#!/usr/bin/env python3
"""Convert an Ultralytics YOLOv8 *detection* checkpoint (.pt) into the safetensors layout used by
stable-diffusion.cpp's ADetailer (same output as vendor/.../scripts/convert_yolov8_to_safetensors.py)
WITHOUT executing code from the checkpoint and without installing ultralytics.

.pt files are pickles. Instead of torch.load(weights_only=False), a restricted unpickler only
resolves PyTorch's tensor-rebuild helpers and plain containers; every other global (Ultralytics and
torch.nn classes, functions, ...) becomes an inert placeholder that just records its state. The
weights are then collected from the module tree and Conv+BatchNorm pairs are fused exactly like
Ultralytics' `fuse()` (fuse_conv_and_bn).

Usage: python -I convert_yolov8_safe.py face_yolov8m.pt face_yolov8m.safetensors
Needs: torch, safetensors.
"""

import argparse
import json
import pickle
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

SAFE_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch", "Size"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
}
DTYPES = {"float16", "float32", "float64", "bfloat16", "int64", "int32", "int8", "uint8", "bool"}


class Inert:
    """Placeholder for any non-allowed class or function found in the pickle."""

    def __init__(self, *args, **kwargs):
        self._inert_args = (args, kwargs)

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
            state = state[0]
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._inert_state = state

    def __call__(self, *args, **kwargs):
        return Inert()


_inert_types = {}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in SAFE_GLOBALS:
            return super().find_class(module, name)
        if module == "torch" and name in DTYPES:
            return getattr(torch, name)
        key = f"{module}.{name}"
        if key not in _inert_types:
            _inert_types[key] = type(name, (Inert,), {"__module__": "inert." + module})
        return _inert_types[key]


restricted_pickle = types.ModuleType("restricted_pickle")
restricted_pickle.Unpickler = RestrictedUnpickler
restricted_pickle.load = lambda f, **kw: RestrictedUnpickler(f, **kw).load()
restricted_pickle.__name__ = "restricted_pickle"


def children(mod):
    return {k: v for k, v in (getattr(mod, "_modules", None) or {}).items() if v is not None}


def collect(mod, prefix, out):
    for k, v in (getattr(mod, "_parameters", None) or {}).items():
        if v is not None:
            out[prefix + k] = v.detach().float()
    for k, v in (getattr(mod, "_buffers", None) or {}).items():
        if v is not None and isinstance(v, torch.Tensor):
            out[prefix + k] = v.detach().float()
    for k, v in children(mod).items():
        collect(v, prefix + k + ".", out)


def fuse(mod, prefix, state):
    """Fold BatchNorm into the preceding conv for every module with `conv` + `bn` children."""
    kids = children(mod)
    if "conv" in kids and "bn" in kids:
        p, b = prefix + "conv.", prefix + "bn."
        w = state[p + "weight"]
        eps = float(getattr(kids["bn"], "eps", 1e-3))
        scale = state[b + "weight"] / torch.sqrt(state[b + "running_var"] + eps)
        bias = state.get(p + "bias", torch.zeros(w.shape[0]))
        state[p + "weight"] = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
        state[p + "bias"] = scale * bias + state[b + "bias"] - scale * state[b + "running_mean"]
        for k in [k for k in state if k.startswith(b)]:
            del state[k]
    for k, v in kids.items():
        fuse(v, prefix + k + ".", state)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--input-size", type=int, default=640)
    args = ap.parse_args()
    if args.input_size < 32 or args.input_size % 32:
        raise SystemExit("--input-size must be a positive multiple of 32")
    if args.output.suffix.lower() != ".safetensors":
        raise SystemExit("output must end in .safetensors")

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False, pickle_module=restricted_pickle)
    if not isinstance(ckpt, dict) or ("model" not in ckpt and "ema" not in ckpt):
        raise SystemExit("not an Ultralytics checkpoint")
    net = ckpt.get("ema") or ckpt.get("model")
    seq = children(net).get("model")
    if seq is None:
        raise SystemExit("unexpected checkpoint layout (no .model)")
    layers = children(seq)
    head = layers[str(len(layers) - 1)]
    if type(head).__name__ != "Detect":
        raise SystemExit(f"only YOLOv8 detection checkpoints are supported (head is {type(head).__name__})")

    state = {}
    collect(net, "", state)
    fuse(net, "", state)

    required = {"model.0.conv.weight", "model.22.cv2.0.2.weight", "model.22.cv3.0.2.weight"}
    missing = sorted(required.difference(state))
    if missing:
        raise SystemExit(f"checkpoint does not match the YOLOv8 layout; missing {missing}")

    tensors = {}
    for name, t in state.items():
        if not name.startswith("model.") or ".bn." in name or name.endswith("dfl.conv.weight"):
            continue
        if not (name.endswith(".weight") or name.endswith(".bias")):
            continue
        dtype = torch.float16 if name.endswith(".weight") else torch.float32
        tensors[name] = t.to(dtype=dtype).contiguous()

    names = getattr(net, "names", None) or getattr(head, "names", None) or {0: "object"}
    if isinstance(names, list):
        names = dict(enumerate(names))
    metadata = {
        "format": "pt",
        "yolov8.variant": "detect",
        "yolov8.input_size": str(args.input_size),
        "yolov8.num_classes": str(int(getattr(head, "nc", len(names)))),
        "yolov8.reg_max": str(int(getattr(head, "reg_max", 16))),
        "yolov8.names": json.dumps({int(k): v for k, v in names.items()}, ensure_ascii=False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    blocked = sorted(k for k in _inert_types)
    print(f"wrote {args.output}: {len(tensors)} tensors, classes={metadata['yolov8.names']}")
    print(f"{len(blocked)} pickle globals were neutralised (not executed), e.g. {blocked[:4]}")


if __name__ == "__main__":
    main()
