#!/usr/bin/env python3
"""Shared compression preset helpers for EMD output.

The application now exposes two primary compression presets:
- zstd: highest compression ratio
- lz4: fastest read/write

The helper keeps the UI, CLI, and EMD writers aligned so the names in the GUI
map to the actual HDF5 filter configuration used on disk.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


try:
    import hdf5plugin
except ImportError as exc:  # pragma: no cover - dependency should be present
    hdf5plugin = None
    _HDF5PLUGIN_IMPORT_ERROR = exc
else:
    _HDF5PLUGIN_IMPORT_ERROR = None


CompressionPreset = Tuple[str, str]


def available_compression_presets() -> list[CompressionPreset]:
    """Return the supported user-facing compression presets."""
    return [
        ("zstd", "High compression (zstd)"),
        ("lz4", "Fast compression (lz4)"),
    ]


def compression_preset_description(preset: str) -> str:
    """Return a short human-readable description for the preset."""
    preset = normalize_compression_preset(preset)
    if preset == "zstd":
        return "Max compression ratio; slower than lz4 on read/write."
    if preset == "lz4":
        return "Much faster read/write; about 60% of zstd's compression ratio."
    if preset == "none":
        return "No compression."
    return "Legacy or custom HDF5 compression."


def normalize_compression_preset(preset: str | None) -> str:
    """Map legacy names to the current preset names."""
    if not preset:
        return "none"

    preset = preset.lower()
    legacy_map = {
        "gzip": "zstd",
        "gzip_1": "zstd",
        "gzip_6": "zstd",
        "gzip_9": "zstd",
        "blosc_zstd": "zstd",
        "blosc_lz4": "lz4",
        "lz4hc": "lz4",
        "none": "none",
    }
    return legacy_map.get(preset, preset)


def compression_ratio_hint(preset: str) -> float:
    """Return a coarse relative compression-ratio hint for UI estimates."""
    preset = normalize_compression_preset(preset)
    if preset == "zstd":
        return 1.0
    if preset == "lz4":
        return 0.60
    return 1.0


def build_hdf5_compression_kwargs(preset: str) -> Dict[str, Any]:
    """Build kwargs for h5py.create_dataset() for the selected preset."""
    preset = normalize_compression_preset(preset)

    if preset == "none":
        return {}

    if preset in {"zstd", "lz4"}:
        if hdf5plugin is None:
            raise RuntimeError(
                "hdf5plugin is required for zstd/lz4 compression. "
                f"Import error: {_HDF5PLUGIN_IMPORT_ERROR}"
            )

        if preset == "zstd":
            return dict(
                hdf5plugin.Blosc(
                    cname="zstd",
                    clevel=3,
                    shuffle=hdf5plugin.Blosc.SHUFFLE,
                )
            )

        return dict(
            hdf5plugin.Blosc(
                cname="lz4",
                clevel=5,
                shuffle=hdf5plugin.Blosc.SHUFFLE,
            )
        )

    # Backward compatibility for legacy built-in HDF5 filters.
    if preset == "lzf":
        return {"compression": "lzf"}
    if preset == "szip":
        return {"compression": "szip"}
    if preset.startswith("gzip"):
        level = 6
        if "_" in preset:
            try:
                level = int(preset.split("_", 1)[1])
            except ValueError:
                pass
        return {"compression": "gzip", "compression_opts": level}

    return {"compression": preset}
