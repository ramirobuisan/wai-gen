import Foundation
import MLX

/// Reads the tensors of a PyTorch checkpoint (`torch.save` zip format, e.g. ESRGAN `.pth` files)
/// without running any pickle code: only plain dictionaries of tensors are understood. Anything else
/// in the pickle (other classes, functions, objects) makes it fail instead of executing it.
public enum TorchCheckpoint {
    public struct Failure: LocalizedError {
        public let message: String
        public var errorDescription: String? { message }
    }

    /// All tensors, keyed by name. Nested dictionaries are flattened with "." (so a file holding
    /// {"params_ema": {...}} gives "params_ema.conv_first.weight", ...).
    public static func load(url: URL) throws -> [String: MLXArray] {
        let zip = try ZipStore(data: Data(contentsOf: url, options: .mappedIfSafe))
        guard let pklName = zip.names.first(where: { $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }) else {
            throw Failure(message: "\(url.lastPathComponent) isn't a PyTorch zip checkpoint")
        }
        let root = String(pklName.dropLast("data.pkl".count))
        var reader = Unpickler(data: try zip.read(pklName))
        let value = try reader.run()
        var out: [String: MLXArray] = [:]
        func walk(_ v: PickleValue, _ prefix: String) throws {
            switch v {
            case .dict(let items):
                for (k, item) in items {
                    guard case .string(let key) = k else { continue }
                    try walk(item, prefix.isEmpty ? key : "\(prefix).\(key)")
                }
            case .tensor(let t):
                out[prefix] = try t.array(storage: { try zip.read("\(root)data/\($0)") })
            default:
                break   // metadata (numbers, strings, ...) is ignored
            }
        }
        try walk(value, "")
        return out
    }

    // MARK: zip (stored entries only, which is what torch.save writes)

    struct ZipStore {
        let data: Data
        private(set) var entries: [String: (offset: Int, size: Int)] = [:]
        var names: [String] { Array(entries.keys) }

        init(data: Data) throws {
            self.data = data
            func u16(_ o: Int) -> Int { Int(data[data.startIndex + o]) | Int(data[data.startIndex + o + 1]) << 8 }
            func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
            // end of central directory record
            var eocd = data.count - 22
            while eocd >= 0, u32(eocd) != 0x0605_4b50 { eocd -= 1 }
            guard eocd >= 0 else { throw Failure(message: "not a zip file") }
            var p = u32(eocd + 16)
            for _ in 0 ..< u16(eocd + 10) {
                guard u32(p) == 0x0201_4b50 else { throw Failure(message: "damaged zip directory") }
                let method = u16(p + 10), size = u32(p + 20), nameLen = u16(p + 28)
                let extraLen = u16(p + 30), commentLen = u16(p + 32), local = u32(p + 42)
                let name = String(decoding: data[(data.startIndex + p + 46) ..< (data.startIndex + p + 46 + nameLen)], as: UTF8.self)
                guard method == 0 else { throw Failure(message: "compressed zip entry \(name) isn't supported") }
                guard size != 0xFFFF_FFFF, local != 0xFFFF_FFFF else { throw Failure(message: "zip64 isn't supported") }
                let start = local + 30 + u16(local + 26) + u16(local + 28)
                entries[name] = (start, size)
                p += 46 + nameLen + extraLen + commentLen
            }
        }

        func read(_ name: String) throws -> Data {
            guard let e = entries[name], e.offset + e.size <= data.count else { throw Failure(message: "missing \(name)") }
            return data[(data.startIndex + e.offset) ..< (data.startIndex + e.offset + e.size)]
        }
    }

    // MARK: pickle

    struct TensorRef {
        let storageKey: String
        let dtype: DType
        let elementSize: Int
        let offset: Int
        let shape: [Int]
        let stride: [Int]

        func array(storage: (String) throws -> Data) throws -> MLXArray {
            var expected = 1
            for (s, n) in zip(stride.reversed(), shape.reversed()) {
                guard n == 1 || s == expected else { throw Failure(message: "non-contiguous tensor isn't supported") }
                expected *= n
            }
            let count = shape.reduce(1, *)
            let bytes = try storage(storageKey)
            let start = offset * elementSize
            guard start + count * elementSize <= bytes.count else { throw Failure(message: "tensor outside its storage") }
            return MLXArray(bytes[(bytes.startIndex + start) ..< (bytes.startIndex + start + count * elementSize)],
                            shape.isEmpty ? [1] : shape, dtype: dtype)
        }
    }

    indirect enum PickleValue {
        case none, bool(Bool), int(Int), float(Double), string(String)
        case tuple([PickleValue]), list([PickleValue]), dict([(PickleValue, PickleValue)])
        case global(String), storage(key: String, dtype: DType, size: Int), tensor(TensorRef)
        case mark
    }

    struct Unpickler {
        let data: Data
        var pos = 0
        var stack: [PickleValue] = []
        var memo: [Int: PickleValue] = [:]

        init(data: Data) { self.data = Data(data) }

        mutating func byte() throws -> UInt8 {
            guard pos < data.count else { throw Failure(message: "truncated pickle") }
            defer { pos += 1 }
            return data[pos]
        }
        mutating func bytes(_ n: Int) throws -> Data {
            guard pos + n <= data.count else { throw Failure(message: "truncated pickle") }
            defer { pos += n }
            return data[pos ..< pos + n]
        }
        mutating func uint(_ n: Int) throws -> Int {
            try bytes(n).reversed().reduce(0) { $0 << 8 | Int($1) }
        }
        mutating func line() throws -> String {
            var s = Data()
            while true { let b = try byte(); if b == 0x0A { break }; s.append(b) }
            return String(decoding: s, as: UTF8.self)
        }
        mutating func pop() throws -> PickleValue {
            guard let v = stack.popLast() else { throw Failure(message: "bad pickle (empty stack)") }
            return v
        }
        mutating func popMark() throws -> [PickleValue] {
            guard let i = stack.lastIndex(where: { if case .mark = $0 { return true } else { return false } }) else {
                throw Failure(message: "bad pickle (no mark)")
            }
            let items = Array(stack[(i + 1)...])
            stack.removeSubrange(i...)
            return items
        }
        static func ints(_ v: PickleValue) -> [Int]? {
            guard case .tuple(let items) = v else { return nil }
            var out: [Int] = []
            for item in items { guard case .int(let i) = item else { return nil }; out.append(i) }
            return out
        }

        mutating func run() throws -> PickleValue {
            while true {
                let op = try byte()
                switch op {
                case 0x80: _ = try byte()                                   // PROTO
                case 0x95: _ = try bytes(8)                                 // FRAME
                case 0x2E: return try pop()                                 // STOP
                case 0x28: stack.append(.mark)                              // MARK
                case 0x7D: stack.append(.dict([]))                          // EMPTY_DICT
                case 0x5D: stack.append(.list([]))                          // EMPTY_LIST
                case 0x29: stack.append(.tuple([]))                         // EMPTY_TUPLE
                case 0x4E: stack.append(.none)                              // NONE
                case 0x88: stack.append(.bool(true))                        // NEWTRUE
                case 0x89: stack.append(.bool(false))                       // NEWFALSE
                case 0x4B: stack.append(.int(try uint(1)))                  // BININT1
                case 0x4D: stack.append(.int(try uint(2)))                  // BININT2
                case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4)))))   // BININT
                case 0x8A:                                                  // LONG1
                    let n = Int(try byte())
                    let raw = try bytes(n)
                    var v = 0
                    for (i, b) in raw.enumerated() { v |= Int(b) << (8 * i) }
                    if n > 0, n < 8, raw.last! & 0x80 != 0 { v -= 1 << (8 * n) }
                    stack.append(.int(v))
                case 0x47:                                                  // BINFLOAT
                    stack.append(.float(Double(bitPattern: UInt64(try uint(8)))))
                case 0x58:                                                  // BINUNICODE
                    let n = try bytes(4).reversed().reduce(0) { $0 << 8 | Int($1) }
                    stack.append(.string(String(decoding: try bytes(n), as: UTF8.self)))
                case 0x8C:                                                  // SHORT_BINUNICODE
                    let n = Int(try byte())
                    stack.append(.string(String(decoding: try bytes(n), as: UTF8.self)))
                case 0x71: memo[Int(try byte())] = stack.last               // BINPUT
                case 0x72: memo[try bytes(4).reversed().reduce(0) { $0 << 8 | Int($1) }] = stack.last   // LONG_BINPUT
                case 0x94: memo[memo.count] = stack.last                    // MEMOIZE
                case 0x68:                                                  // BINGET
                    guard let v = memo[Int(try byte())] else { throw Failure(message: "bad memo") }
                    stack.append(v)
                case 0x6A:                                                  // LONG_BINGET
                    guard let v = memo[try bytes(4).reversed().reduce(0) { $0 << 8 | Int($1) }] else {
                        throw Failure(message: "bad memo")
                    }
                    stack.append(v)
                case 0x74: stack.append(.tuple(try popMark()))              // TUPLE
                case 0x85: stack.append(.tuple([try pop()]))                // TUPLE1
                case 0x86: let b = try pop(), a = try pop(); stack.append(.tuple([a, b]))           // TUPLE2
                case 0x87: let c = try pop(), b = try pop(), a = try pop(); stack.append(.tuple([a, b, c]))   // TUPLE3
                case 0x61:                                                  // APPEND
                    let v = try pop()
                    guard case .list(var l) = try pop() else { throw Failure(message: "bad append") }
                    l.append(v); stack.append(.list(l))
                case 0x65:                                                  // APPENDS
                    let items = try popMark()
                    guard case .list(var l) = try pop() else { throw Failure(message: "bad appends") }
                    l += items; stack.append(.list(l))
                case 0x73:                                                  // SETITEM
                    let v = try pop(), k = try pop()
                    guard case .dict(var d) = try pop() else { throw Failure(message: "bad setitem") }
                    d.append((k, v)); stack.append(.dict(d))
                case 0x75:                                                  // SETITEMS
                    let items = try popMark()
                    guard case .dict(var d) = try pop(), items.count % 2 == 0 else { throw Failure(message: "bad setitems") }
                    for i in stride(from: 0, to: items.count, by: 2) { d.append((items[i], items[i + 1])) }
                    stack.append(.dict(d))
                case 0x63:                                                  // GLOBAL "module\nname\n"
                    let module = try line(), name = try line()
                    stack.append(.global("\(module).\(name)"))
                case 0x93:                                                  // STACK_GLOBAL
                    guard case .string(let name) = try pop(), case .string(let module) = try pop() else {
                        throw Failure(message: "bad global")
                    }
                    stack.append(.global("\(module).\(name)"))
                case 0x51:                                                  // BINPERSID
                    // ('storage', <storage class>, key, location, numel)
                    guard case .tuple(let t) = try pop(), t.count >= 5, case .global(let cls) = t[1],
                          case .string(let key) = t[2], case .int(let size) = t[4]
                    else { throw Failure(message: "unsupported persistent id") }
                    let types: [String: DType] = ["torch.FloatStorage": .float32, "torch.HalfStorage": .float16,
                                                  "torch.BFloat16Storage": .bfloat16]
                    guard let dtype = types[cls] else { throw Failure(message: "unsupported storage \(cls)") }
                    stack.append(.storage(key: key, dtype: dtype, size: size))
                case 0x52:                                                  // REDUCE
                    let args = try pop()
                    guard case .global(let fn) = try pop() else { throw Failure(message: "bad reduce") }
                    switch fn {
                    case "collections.OrderedDict":
                        stack.append(.dict([]))
                    case "torch._utils._rebuild_tensor_v2":
                        guard case .tuple(let a) = args, a.count >= 4, case .storage(let key, let dtype, _) = a[0],
                              case .int(let offset) = a[1], let shape = Self.ints(a[2]), let stride = Self.ints(a[3])
                        else { throw Failure(message: "bad tensor") }
                        let size = dtype == .float32 ? 4 : 2
                        stack.append(.tensor(TensorRef(storageKey: key, dtype: dtype, elementSize: size, offset: offset,
                                                       shape: shape, stride: stride)))
                    case "torch._utils._rebuild_parameter":
                        guard case .tuple(let a) = args, let t = a.first else { throw Failure(message: "bad parameter") }
                        stack.append(t)
                    default:
                        throw Failure(message: "refusing to call \(fn) from a checkpoint")
                    }
                case 0x62: _ = try pop()                                    // BUILD: drop the state (OrderedDict metadata)
                default:
                    throw Failure(message: String(format: "unsupported pickle opcode 0x%02X", op))
                }
            }
        }
    }
}
