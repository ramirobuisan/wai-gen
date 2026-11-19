// swift-tools-version: 5.10
// WAI Gen — native macOS image generator for the WAI models (no Electron, no ComfyUI).

import Foundation
import PackageDescription

// Prebuilt stable-diffusion.cpp (Metal) static libraries, see vendor/stable-diffusion.cpp/build.
let sdcppBuild = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .appendingPathComponent("vendor/stable-diffusion.cpp/build").path

let package = Package(
    name: "wai-gen",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "WaiMLX", targets: ["WaiMLX"]),
        .library(name: "WaiSDCpp", targets: ["WaiSDCpp"]),
        .executable(name: "waigen-cli", targets: ["waigen-cli"]),
        .executable(name: "WAIGen", targets: ["WAIGen"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
    ],
    targets: [
        // SDXL engine on MLX. Model code (Clip, UNet, VAE, Tokenizer, Image, Configuration) is
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
        // C module over the vendored stable-diffusion.cpp API (used for WAI-ANIMA).
        .target(
            name: "CStableDiffusion",
            linkerSettings: [
                .unsafeFlags([
                    "-L\(sdcppBuild)", "-L\(sdcppBuild)/ggml/src",
                    "-L\(sdcppBuild)/ggml/src/ggml-metal", "-L\(sdcppBuild)/ggml/src/ggml-blas",
                    "-L\(sdcppBuild)/thirdparty/libwebp", "-L\(sdcppBuild)/thirdparty/libwebm",
                ]),
                .linkedLibrary("stable-diffusion"), .linkedLibrary("ggml"),
                .linkedLibrary("ggml-base"), .linkedLibrary("ggml-cpu"),
                .linkedLibrary("ggml-metal"), .linkedLibrary("ggml-blas"),
                .linkedLibrary("webpmux"), .linkedLibrary("webp"), .linkedLibrary("sharpyuv"),
                .linkedLibrary("webm"), .linkedLibrary("c++"),
                .linkedFramework("Metal"), .linkedFramework("MetalKit"),
                .linkedFramework("Foundation"), .linkedFramework("Accelerate"),
            ]
        ),
        .target(name: "WaiSDCpp", dependencies: ["CStableDiffusion"]),
        .executableTarget(name: "waigen-cli", dependencies: ["WaiMLX", "WaiSDCpp"]),
        .executableTarget(name: "WAIGen", dependencies: ["WaiMLX", "WaiSDCpp"]),
    ]
)
