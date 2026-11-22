import SwiftUI

struct InspectorView: View {
    @Environment(AppModel.self) private var model

    // collapsible sections (remembered)
    @AppStorage("inspector.openImage") private var openImage = true
    @AppStorage("inspector.openHires") private var openHires = true
    @AppStorage("inspector.openLoRA") private var openLoRA = true
    @AppStorage("inspector.openFaceRepair") private var openFaceRepair = true
    @AppStorage("inspector.openPrompt") private var openPrompt = true

    private static let sizes: [(Int, Int)] = [(1024, 1344), (832, 1216), (896, 1152), (1024, 1024), (1152, 896), (1216, 832), (1344, 1024)]

    var body: some View {
        @Bindable var model = model
        let preset = model.preset
        let missing = model.missingModelFiles

        Form {
            Section {
                if model.isRemote {
                    if let info = model.remoteInfo {
                        Label("Generating on \(info.name)", systemImage: "network")
                            .foregroundStyle(.blue)
                            .help("Settings ▸ Generator chooses where images are made")
                        if !info.loadedModels.isEmpty {
                            Label("In memory there: \(info.loadedModels.joined(separator: ", "))", systemImage: "memorychip")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    } else {
                        HStack {
                            Label(model.remoteStatus ?? "Not connected to \(model.generatorName)", systemImage: "network.slash")
                                .foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                            Spacer()
                            Button("Retry") { model.connectGenerator() }
                        }
                    }
                }
                if let client = model.servingClient {
                    Label("Generating for \(client)", systemImage: "arrow.down.left.and.arrow.up.right")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !missing.isEmpty {
                    Label(model.isRemote ? "Not available on \(model.generatorName)" : "Model files missing",
                          systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .help(missing.joined(separator: "\n"))
                } else if model.settings.matches(preset) {
                    Label("Preset applied · \(preset.engine.rawValue)", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .help(preset.source)
                } else {
                    HStack {
                        Label("Changed from preset", systemImage: "slider.horizontal.3").foregroundStyle(.secondary)
                        Spacer()
                        Button("Reset") { model.resetToPreset() }
                    }
                    .help(preset.source)
                }
                if model.isRemote {
                    EmptyView()
                } else if let loading = model.backgroundLoading {
                    Label("Loading \(loading) in the background…", systemImage: "arrow.down.circle")
                        .font(.caption).foregroundStyle(.secondary)
                } else if !model.loadedModels.isEmpty {
                    Label("In memory: \(model.loadedModels.joined(separator: ", "))", systemImage: "memorychip")
                        .font(.caption).foregroundStyle(.secondary)
                        .help("Model ▸ Unload All Models frees the memory")
                }
            }

            CollapsibleSection(title: "Image", isExpanded: $openImage) {
                Picker("Size", selection: Binding(
                    get: { "\(model.settings.width)x\(model.settings.height)" },
                    set: { v in
                        let p = v.split(separator: "x").compactMap { Int($0) }
                        if p.count == 2 { model.settings.width = p[0]; model.settings.height = p[1] }
                    })) {
                    // id must be the whole size: 1024×1344 and 1024×1024 share a width
                    ForEach(Self.sizes.map { "\($0.0)x\($0.1)" }, id: \.self) { key in
                        Text(verbatim: key.replacingOccurrences(of: "x", with: " × ")).tag(key)
                    }
                    if !Self.sizes.contains(where: { $0 == (model.settings.width, model.settings.height) }) {
                        Text(verbatim: "\(model.settings.width) × \(model.settings.height)").tag("\(model.settings.width)x\(model.settings.height)")
                    }
                }
                SliderRow(title: "Steps", value: Binding(get: { Double(model.settings.steps) }, set: { model.settings.steps = Int($0) }),
                          range: 8 ... 60, step: 1, format: "%.0f")
                SliderRow(title: "CFG", value: $model.settings.cfg, range: 1 ... 12, step: 0.5, format: "%.1f")
                Picker("Sampler", selection: $model.settings.sampler) {
                    ForEach(Sampler.allCases) { Text($0.label).tag($0) }
                }
                Picker("Schedule", selection: $model.settings.scheduler) {
                    ForEach(Scheduler.allCases) { Text($0.label).tag($0) }
                }
                SeedRow(seed: $model.settings.seed)
                Stepper("Images: \(model.settings.batchCount)", value: $model.settings.batchCount, in: 1 ... 16)
                Toggle("Live preview", isOn: $model.settings.livePreview)
                    .help("Shows the image forming at every step. It's computed together with each step, so it doesn't slow generation down.")
                VStack(alignment: .leading, spacing: 2) {
                    Toggle("Fast mode", isOn: $model.settings.fastMode)
                    Text("\(preset.defaults.fast.sampler.label) · \(preset.defaults.fast.scheduler.label) · \(preset.defaults.fast.steps) steps: about 30 % faster, a slightly different look")
                        .font(.caption).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Toggle("Step cache", isOn: $model.settings.stepCache)
                    Text("Reuses the model's output on steps where it barely changes: about 25 % faster, nearly the same image")
                        .font(.caption).foregroundStyle(.secondary)
                    let sampler = model.settings.fastMode ? preset.defaults.fast.sampler : model.settings.sampler
                    if model.settings.stepCache, !sampler.supportsStepCache {
                        Text("Not used with \(sampler.label)\(model.settings.fastMode ? " (Fast mode)" : ""): it only works with Euler a; with other samplers it skips too much and the image gets grainy")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
            }

            CollapsibleSection(title: "Hi-res fix", isExpanded: $openHires) {
                Toggle("Enabled", isOn: $model.settings.hiresEnabled)
                if model.settings.hiresEnabled {
                    SliderRow(title: "Scale", value: $model.settings.hiresScale,
                              range: 1.1 ... preset.defaults.maxHiresScale, step: 0.05, format: "%.2f×")
                    SliderRow(title: "Steps", value: Binding(get: { Double(model.settings.hiresSteps) }, set: { model.settings.hiresSteps = Int($0) }),
                              range: 5 ... 40, step: 1, format: "%.0f")
                    SliderRow(title: "Denoise", value: $model.settings.hiresDenoise, range: 0.1 ... 0.8, step: 0.05, format: "%.2f")
                    if preset.engine == .sdxl {
                        VStack(alignment: .leading, spacing: 2) {
                            Toggle("Faster hi-res guidance", isOn: $model.settings.fastHiresGuidance)
                            Text("Computes the guidance on every other hi-res step and reuses it in between: about 10 % faster per image, nearly the same image")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        .help("Also used by the gallery's Upscale with detail pass. WAI v16 only.")
                    }
                    Picker("Upscaler", selection: $model.settings.hiresUpscaler) {
                        Text(preset.engine == .sdxl ? "Built-in (bicubic)" : "Built-in (high-quality resize)").tag("")
                        ForEach(model.upscalerFiles, id: \.name) { f in
                            Text("\((f.name as NSString).deletingPathExtension) ×\(f.scale)").tag(f.name)
                        }
                        if !model.settings.hiresUpscaler.isEmpty,
                            !model.upscalerFiles.contains(where: { $0.name == model.settings.hiresUpscaler }) {
                            Text("\(model.settings.hiresUpscaler) (not found)").tag(model.settings.hiresUpscaler)
                        }
                    }
                    .help("ESRGAN-type models in models/upscale_models. WAI's authors use R-ESRGAN 4x+ Anime6B.")
                    Text(verbatim: "\(Int(Double(model.settings.width) * model.settings.hiresScale / 8) * 8) × \(Int(Double(model.settings.height) * model.settings.hiresScale / 8) * 8) final")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            CollapsibleSection(title: "LoRA", isExpanded: $openLoRA) {
                ForEach($model.settings.loras) { $sel in
                    let file = model.loraFiles.first { $0.relativePath == sel.relativePath }
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Toggle(isOn: $sel.enabled) { Text(sel.name).lineLimit(1).truncationMode(.middle) }
                                .toggleStyle(.checkbox)
                            Spacer(minLength: 0)
                            if file == nil {
                                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                                    .help("Not in the loras folder anymore. Remove it to generate.")
                            } else if let file, file.fits(preset.engine) == false {
                                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                                    .help("Made for \(file.baseModel ?? "another model"), not \(preset.name). Skipped when generating.")
                            }
                            if let tags = file?.triggerTags, !tags.isEmpty {
                                Button {
                                    let add = tags.prefix(4).joined(separator: ", ")
                                    let p = model.settings.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                                    model.settings.prompt = p.isEmpty ? add : p + (p.hasSuffix(",") ? " " : ", ") + add
                                } label: { Label("Add Trigger Words", systemImage: "text.badge.plus").labelStyle(.iconOnly) }
                                    .buttonStyle(.borderless)
                                    .help("Add trigger words: \(tags.prefix(4).joined(separator: ", "))")
                            }
                            Button { model.settings.loras.removeAll { $0.id == sel.id } } label: {
                                Label("Remove LoRA", systemImage: "minus.circle").labelStyle(.iconOnly)
                            }
                            .buttonStyle(.borderless)
                            .help("Remove")
                        }
                        SliderRow(title: "Strength", value: $sel.strength, range: -1 ... 2, step: 0.05, format: "%.2f")
                    }
                }
                Menu("Add LoRA") {
                    if model.loraFiles.isEmpty {
                        Text("No LoRAs in \(LoRACatalog.folder.path)")
                    }
                    ForEach(model.loraFiles) { f in
                        Button(f.baseModel.map { "\(f.name)  ·  \($0)" } ?? f.name) { model.addLoRA(f) }
                            .disabled(model.settings.loras.contains { $0.relativePath == f.relativePath })
                    }
                    Divider()
                    Button("Refresh list") { model.refreshLoRAs() }
                    Button("Open LoRA folder") { NSWorkspace.shared.open(LoRACatalog.folder) }
                }
            }

            CollapsibleSection(title: "Face repair", isExpanded: $openFaceRepair) {
                Toggle("Enabled", isOn: $model.settings.faceRepair)
                    .help("Finds faces with a YOLOv8 detector and repaints each one at 1024×1024, like ADetailer")
                if model.settings.faceRepair {
                    if !FileManager.default.fileExists(atPath: AppPaths.faceDetector.path) {
                        Label("Face detector missing", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .help("Expected at \(AppPaths.faceDetector.path)")
                    }
                    SliderRow(title: "Strength", value: $model.settings.faceStrength, range: 0.2 ... 0.7, step: 0.05, format: "%.2f")
                    SliderRow(title: "Detection confidence", value: $model.settings.faceConfidence, range: 0.1 ... 0.9, step: 0.05, format: "%.2f")
                    TextField("Face prompt", text: $model.settings.facePrompt, prompt: Text("Same as the prompt"), axis: .vertical)
                        .lineLimit(1 ... 3)
                        .help("Optional extra tags for the faces, e.g. \"detailed eyes, smile\". Characters and quality tags are added automatically.")
                }
            }

            CollapsibleSection(title: "Prompt", isExpanded: $openPrompt) {
                Toggle("Quality tags", isOn: $model.settings.useQualityPrefix)
                if model.settings.useQualityPrefix {
                    TextField("Quality tags", text: $model.settings.qualityPrefix, axis: .vertical)
                        .font(.caption).lineLimit(1 ... 3)
                }
                Toggle("Tag assist", isOn: $model.settings.tagAssist)
                    .help("Adds the extra tags SAA's dataset lists for some characters")
                let wildcards = Wildcards.available()
                if !wildcards.isEmpty {
                    LabeledContent("Wildcards") {
                        Text(wildcards.map { "__\($0)__" }.joined(separator: " "))
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
            }
        }
        .formStyle(.grouped)
        // the model picker lives in the toolbar strip above the inspector (that strip was empty),
        // and the settings start right under it
        .contentMargins(.top, 0, for: .scrollContent)
        .toolbar {
            ToolbarItem {
                // a toolbar Picker only shows its arrows; a Menu shows the model's name
                Menu {
                    Picker("Model", selection: Binding(get: { preset }, set: { model.selectModel($0) })) {
                        ForEach(ModelCatalog.presets) { Text($0.name).tag($0) }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                } label: {
                    Text(preset.name)
                }
                .menuIndicator(.visible)
                .fixedSize()
                .help("Model (⌘1 / ⌘2 / ⌘3)")
            }
        }
    }
}

struct SliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: Binding(get: { min(max(value, range.lowerBound), range.upperBound) }, set: { value = $0 }),
                   in: range, step: step)
                .labelsHidden()
        }
    }
}

struct SeedRow: View {
    @Binding var seed: Int64
    @State private var text = ""

    var body: some View {
        HStack {
            Text("Seed")
            Spacer()
            // in a Form the title shows as a label: hide it, "Random" is only the empty field's placeholder
            TextField("Seed", text: $text, prompt: Text("Random"))
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 110)
                // applies as you type (Generate right after typing uses it); empty = random
                .onChange(of: text) { _, t in
                    let s = t.trimmingCharacters(in: .whitespaces)
                    if s.isEmpty {
                        if seed != -1 { seed = -1 }
                    } else if let v = Int64(s), v >= 0, v != seed {
                        seed = v
                    }
                }
                .onSubmit { text = seed < 0 ? "" : String(seed) }   // drops anything that isn't a number
            Button { seed = -1 } label: { Label("Random Seed", systemImage: "dice").labelStyle(.iconOnly) }
                .buttonStyle(.borderless)
                .help("Random seed for every image")
        }
        .onAppear { text = seed < 0 ? "" : String(seed) }
        .onChange(of: seed) { _, v in text = v < 0 ? "" : String(v) }
    }
}

struct ImageInfoView: View {
    @Environment(AppModel.self) private var model
    let item: GalleryItem

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let r = item.record {
                Text(r.modelName).font(.headline)
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    // an upscaled image's record holds the final size; "Upscaled from" below says the rest
                    row("Size", "\(r.width) × \(r.height)" + (r.upscaledFrom != nil ? ""
                        : r.hiresScale.map { " → hi-res \(String(format: "%.2f", $0))×" } ?? ""))
                    row("Sampler", "\(r.sampler) · \(r.scheduler) · \(r.steps) steps · CFG \(String(format: "%.1f", r.cfg))")
                    row("Seed", "\(r.seed)")
                    row("Time", String(format: "%.1f s · %@", r.seconds, r.engine))
                    if let src = r.upscaledFrom {
                        row("Upscaled from", src + (r.upscaleDetailPass == false ? " (upscaler only)" : " (with detail pass)"))
                    }
                    if let fs = r.faceRepairStrength {
                        row("Face repair", String(format: "strength %.2f · confidence %.2f%@", fs, r.faceRepairConfidence ?? 0.3,
                                                  r.faceRepairSeconds.map { String(format: " · %.1f s", $0) } ?? ""))
                    }
                }
                .font(.callout)
                Text("Prompt").font(.caption).foregroundStyle(.secondary)
                Text(r.positive).font(.callout).textSelection(.enabled)
                Text("Negative").font(.caption).foregroundStyle(.secondary)
                Text(r.negative).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                HStack {
                    Button("Use settings") { model.reuse(item, seed: false) }
                    Button("Use with seed") { model.reuse(item, seed: true) }
                    Spacer()
                    Button("Copy parameters") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(r.parametersText, forType: .string)
                    }
                }
                .padding(.top, 4)
            } else {
                Text(item.url.lastPathComponent).font(.headline)
                Text("No WAI Gen information in this image.").foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(width: 420)
    }

    @ViewBuilder func row(_ k: String, _ v: String) -> some View {
        GridRow {
            Text(k).foregroundStyle(.secondary)
            Text(v).textSelection(.enabled)
        }
    }
}

/// A form section whose whole title row (not only the disclosure arrow) collapses and expands it.
struct CollapsibleSection<Content: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    @ViewBuilder let content: () -> Content

    var body: some View {
        Section(isExpanded: $isExpanded) {
            content()
        } header: {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() }
            } label: {
                Text(title)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityHint(isExpanded ? "Collapses the section" : "Expands the section")
        }
    }
}
