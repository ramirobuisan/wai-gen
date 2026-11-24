import Foundation
import MLX

/// Z-Image's text conditioning: Qwen3-4B through ComfyUI's chat template (the user turn
/// `<|im_start|>user\n`, then the assistant-turn opener `<|im_end|>\n<|im_start|>assistant\n`,
/// with the Qwen3 hybrid thinking block left empty, as in ComfyUI's `thinking=True`), using the
/// hidden states of the second-to-last decoder layer (`layer_idx = -2`) without the final norm —
/// exactly what `Qwen3_4BModel(layer="hidden", layer_idx=-2)` in `comfy/text_encoders/z_image.py` computes.
public final class ZImageTextEncoder {
    public let encoder: Qwen3TextEncoder
    private let tokenizer: Qwen2Tokenizer

    // ComfyUI's Qwen3 chat special tokens.
    private static let imStart = "<|im_start|>"
    private static let imEnd = "<|im_end|>"

    // ComfyUI's `llama_template` with `thinking=True`: imStart + "user\n" + prompt + imEnd + "\n" + imStart + "assistant\n"
    private static let prefix = imStart + "user\n"
    private static let suffix = imEnd + "\n" + imStart + "assistant\n"

    public init(encoder: Qwen3TextEncoder, tokenizer: Qwen2Tokenizer) {
        self.encoder = encoder
        self.tokenizer = tokenizer
        encoder.outputLayerIndex = encoder.layers - 2
    }

    /// The prompt wrapped in the template, tokenized (`<|im_start|>` / `<|im_end|>` are single ids).
    public func tokenize(_ text: String) -> [Int32] {
        tokenizer.encode(Self.prefix + text + Self.suffix)
    }

    /// The template's token ids run through the encoder to the second-to-last layer, no final norm:
    /// [n, hidden] float32.
    public func encode(_ text: String) -> MLXArray {
        let hidden = encoder(tokenize(text))
        eval(hidden)
        return hidden
    }
}
