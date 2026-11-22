// swift-tools-version: 5.10
// WAI Gen — native macOS image generator for the WAI models (no Electron, no ComfyUI).

import Foundation
import PackageDescription

let package = Package(
    name: "wai-gen",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "WaiMLX", targets: ["WaiMLX"]),
        .library(name: "WaiRemote", targets: ["WaiRemote"]),
        .executable(name: "waigen-cli", targets: ["waigen-cli"]),
        .executable(name: "WAIGen", targets: ["WAIGen"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.33.0")),
    ],
    targets: [
        // Both pipelines in MLX: WAI-illustrious SDXL (WaiSDXL) and WAI-ANIMA (WaiAnima). SDXL model code (Clip, UNet, VAE, Tokenizer, Image, Configuration) is
        // Apple's mlx-swift-examples StableDiffusion library (MIT, see THIRD_PARTY_LICENSE_*).
        .target(
            name: "WaiMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            resources: [.copy("Resources")]
        ),
        // Generating on another Mac: WebSocket protocol, server, client, Bonjour discovery (Network.framework).
        .target(name: "WaiRemote"),
        .executableTarget(name: "waigen-cli", dependencies: ["WaiMLX", "WaiRemote"]),
        .executableTarget(name: "WAIGen", dependencies: ["WaiMLX", "WaiRemote"]),
    ]
)
