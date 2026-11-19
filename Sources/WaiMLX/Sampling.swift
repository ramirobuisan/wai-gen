// k-diffusion style sampling for eps-prediction SDXL models (WAI-illustrious), following
// ComfyUI's ModelSamplingDiscrete + EPS (comfy/model_sampling.py) and its k-diffusion samplers.

import Foundation
import MLX
import MLXRandom

public enum SamplerKind: String, CaseIterable, Codable, Sendable {
    case euler
    case eulerAncestral = "euler_ancestral"
    case dpmpp2m = "dpmpp_2m"
}

public enum SchedulerKind: String, CaseIterable, Codable, Sendable {
    case normal
    case karras
    case simple
}

/// The 1000-step discrete noise schedule of SD/SDXL (scaled_linear betas 0.00085 ... 0.012).
struct DiscreteSchedule {
    let sigmas: [Float]       // index = training timestep 0 ... 999
    let logSigmas: [Float]

    init(betaStart: Double = 0.00085, betaEnd: Double = 0.012, steps: Int = 1000) {
        var acp = 1.0
        var s = [Float]()
        let a = betaStart.squareRoot()
        let b = betaEnd.squareRoot()
        for i in 0 ..< steps {
            let beta = pow(a + (b - a) * Double(i) / Double(steps - 1), 2)
            acp *= 1 - beta
            s.append(Float(((1 - acp) / acp).squareRoot()))
        }
        sigmas = s
        logSigmas = s.map { log($0) }
    }

    var sigmaMin: Float { sigmas[0] }
    var sigmaMax: Float { sigmas[sigmas.count - 1] }

    /// sigma -> continuous timestep (interpolated in log-sigma space)
    func timestep(_ sigma: Float) -> Float {
        let ls = log(sigma)
        if ls <= logSigmas[0] { return 0 }
        if ls >= logSigmas[logSigmas.count - 1] { return Float(logSigmas.count - 1) }
        var lo = 0
        var hi = logSigmas.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if logSigmas[mid] <= ls { lo = mid } else { hi = mid }
        }
        let w = (ls - logSigmas[lo]) / (logSigmas[hi] - logSigmas[lo])
        return Float(lo) + w
    }

    /// continuous timestep -> sigma
    func sigma(_ t: Float) -> Float {
        let t = min(max(t, 0), Float(sigmas.count - 1))
        let lo = Int(t.rounded(.down))
        let hi = min(lo + 1, sigmas.count - 1)
        let w = t - Float(lo)
        return exp((1 - w) * logSigmas[lo] + w * logSigmas[hi])
    }

    /// Sigmas for `steps` steps, ending with 0.
    func sigmas(steps: Int, scheduler: SchedulerKind) -> [Float] {
        switch scheduler {
        case .normal:
            let start = timestep(sigmaMax)
            let end = timestep(sigmaMin)
            var out = (0 ..< steps).map { i -> Float in
                let t = steps == 1 ? start : start + (end - start) * Float(i) / Float(steps - 1)
                return sigma(t)
            }
            out.append(0)
            return out
        case .karras:
            let rho: Float = 7
            let minInv = pow(sigmaMin, 1 / rho)
            let maxInv = pow(sigmaMax, 1 / rho)
            var out = (0 ..< steps).map { i -> Float in
                let ramp = steps == 1 ? 0 : Float(i) / Float(steps - 1)
                return pow(maxInv + ramp * (minInv - maxInv), rho)
            }
            out.append(0)
            return out
        case .simple:
            let ss = Float(sigmas.count) / Float(steps)
            var out = (0 ..< steps).map { i in sigmas[sigmas.count - 1 - Int(Float(i) * ss)] }
            out.append(0)
            return out
        }
    }
}

/// Denoiser: given the noisy latent `x` at `sigma`, returns the predicted clean latent.
typealias Denoiser = (_ x: MLXArray, _ sigma: Float) -> MLXArray

/// Run a k-diffusion sampler over `sigmas` (descending, last = 0). `x` must already be noised
/// for `sigmas[0]`. Calls `progress(step, total)` after each step. Returns nil when
/// `isCancelled()` turns true (checked before every step).
func sample(
    _ kind: SamplerKind, denoise: Denoiser, x: MLXArray, sigmas: [Float],
    isCancelled: () -> Bool = { false },
    preview: StepPreview? = nil,
    progress: (Int, Int) -> Void
) -> MLXArray? {
    var x = x
    let total = sigmas.count - 1
    var oldDenoised: MLXArray? = nil

    for i in 0 ..< total {
        if isCancelled() { return nil }
        let s = sigmas[i]
        let sNext = sigmas[i + 1]
        let denoised = denoise(x, s)

        switch kind {
        case .euler:
            let d = (x - denoised) / s
            x = x + d * (sNext - s)

        case .eulerAncestral:
            if sNext == 0 {
                x = denoised
            } else {
                let sUp = min(sNext, (sNext * sNext * (s * s - sNext * sNext) / (s * s)).squareRoot())
                let sDown = (sNext * sNext - sUp * sUp).squareRoot()
                let d = (x - denoised) / s
                x = x + d * (sDown - s)
                x = x + MLXRandom.normal(x.shape).asType(x.dtype) * sUp
            }

        case .dpmpp2m:
            if sNext == 0 {
                x = denoised
            } else {
                let t = -log(s)
                let tNext = -log(sNext)
                let h = tNext - t
                let ratio = sNext / s
                let coef = -expm1(-h)
                if let old = oldDenoised, i > 0 {
                    let hLast = t - (-log(sigmas[i - 1]))
                    let r = hLast / h
                    let d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old
                    x = ratio * x + coef * d
                } else {
                    x = ratio * x + coef * denoised
                }
            }
            oldDenoised = denoised
        }

        if let preview {
            // build the preview from this step's denoised prediction and evaluate it in the same
            // graph as the step, so it costs no extra GPU round trip
            let p = preview.make(denoised)
            eval(x, p)
            preview.deliver(p)
        } else {
            eval(x)
        }
        progress(i + 1, total)
    }
    return x
}

/// Per-step preview: `make` turns the denoised latent into a small image array (lazily, on the GPU),
/// `deliver` receives it after the step has been evaluated.
struct StepPreview {
    let make: (MLXArray) -> MLXArray
    let deliver: (MLXArray) -> Void
}

/// Step cache ("EasyCache", the rule stable-diffusion.cpp uses for its `easycache` mode): between two
/// steps the model's output moves roughly in proportion to its input, at a rate measured on the steps
/// that are computed. While the predicted relative change of the output, accumulated since the last
/// computed step, stays under `threshold`, the step reuses the last output's offset from its input
/// (output ≈ input + (lastOutput - lastInput)) instead of running the model. Only steps between
/// `start` and `end` (fractions of the schedule) can be skipped; the first and last ones always run.
final class StepCache {
    let threshold: Float
    private let startSigma: Float
    private let endSigma: Float
    private var lastInput: MLXArray?
    private var lastOutput: MLXArray?
    private var offset: MLXArray?
    private var rate: Float?
    private var outputNorm: Float = 0
    private var accumulated: Float = 0
    private(set) var skipped = 0

    init(threshold: Float, sigmas: [Float], start: Float = 0.15, end: Float = 0.95) {
        self.threshold = threshold
        let n = max(sigmas.count - 1, 1)
        startSigma = sigmas[min(Int(start * Float(n)), n - 1)]
        endSigma = sigmas[min(Int(end * Float(n)), n - 1)]
    }

    /// `input`: the model input for this step (float32); `compute` runs the model on it.
    func callAsFunction(_ input: MLXArray, sigma: Float, compute: () -> MLXArray) -> MLXArray {
        guard sigma <= startSigma, sigma > endSigma else { return compute() }
        var inputChange: Float? = nil
        if let lastInput {
            let change = abs(input - lastInput).mean().item(Float.self)
            inputChange = change
            if let rate, let offset, outputNorm > 0 {
                accumulated += rate * change / outputNorm
                if accumulated < threshold {
                    skipped += 1
                    return input + offset
                }
            }
        }
        let output = compute().asType(.float32)
        let outputChange = lastOutput.map { abs(output - $0).mean() }
        let norm = abs(output).mean()
        eval(output, norm)
        if let inputChange, inputChange > 0, let oc = outputChange?.item(Float.self), oc > 0, oc.isFinite {
            rate = oc / inputChange
        }
        outputNorm = norm.item(Float.self)
        lastInput = input
        lastOutput = output
        offset = output - input
        accumulated = 0
        return output
    }
}
