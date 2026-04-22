#!/usr/bin/env python3
from __future__ import annotations

"""Export Quantum Detectors MIB files to NumPy arrays plus JSON metadata.

This module is meant to be reusable from the GUI and from scripts. It keeps
the export logic separate from the one-off handoff notebook/script so the
feature can live in the application proper.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .mib_loader import (
    auto_detect_scan_size,
    apply_data_processing,
    calculate_processed_size,
    get_mib_properties,
)


HEADER_SIZE_BYTES = 384
ProgressCallback = Callable[[int, str], None]


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/Python values into JSON-serializable data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_header(input_path: Path) -> tuple[list[str], int]:
    """Read the fixed-size MIB header and return header fields plus file size."""
    with input_path.open("rb") as f:
        header_bytes = f.read(HEADER_SIZE_BYTES)
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

    if len(header_bytes) < HEADER_SIZE_BYTES:
        raise ValueError(
            f"File is too small to be a valid MIB file: {input_path} "
            f"({len(header_bytes)} bytes read)"
        )

    header_fields = header_bytes.decode("utf-8", errors="ignore").split(",")
    return header_fields, file_size


def _build_metadata(
    input_path: Path,
    header_fields: list[str],
    file_size: int,
    props: Any,
    num_frames: int,
    scan_size: tuple[int, int],
    output_shape: tuple[int, int, int, int],
    output_path: Path,
    processing_options: dict[str, Any],
) -> dict[str, Any]:
    """Assemble metadata describing the export."""
    return {
        "source_file": str(input_path),
        "source_filename": input_path.name,
        "source_size_bytes": file_size,
        "source_size_gb": file_size / (1024**3),
        "output_npy": str(output_path),
        "output_json": str(output_path.with_suffix(".json")),
        "header_size_bytes": HEADER_SIZE_BYTES,
        "num_frames": num_frames,
        "scan_size_detected": scan_size,
        "array_shape": output_shape,
        "detector_size_header": tuple(props.merlin_size),
        "detector_mode": "Single" if props.single else "Quad",
        "raw_format": bool(props.raw),
        "pixel_dtype": str(props.pixeltype),
        "dynamic_range": props.dyn_range,
        "processing_options": _jsonable(processing_options),
        "header_fields": header_fields,
        "mib_properties": {
            "path": props.path,
            "scan_size": props.scan_size,
            "xy": props.xy,
            "merlin_size": props.merlin_size,
            "headsize": props.headsize,
            "pixeltype": str(props.pixeltype),
            "offset": props.offset,
            "raw": props.raw,
            "quad": props.quad,
            "single": props.single,
            "dyn_range": props.dyn_range,
            "numberOfFramesInFile": props.numberOfFramesInFile,
            "detectorgeometry": props.detectorgeometry,
        },
        "export_notes": [
            "Output array follows the same orientation rules used by the application's load_mib() logic.",
            "If the detector appears scrambled in EELS mode, the export swaps detector axes the same way the application does.",
            "The energy axis is reversed so zero appears on the left in the viewer convention.",
            "EELS exports default to Y-summing so the NumPy output is a compact spectrum image.",
            "Binning is available for both EELS and 4D STEM exports, but it is off by default.",
        ],
    }


def _default_processing_options(props: Any) -> dict[str, Any]:
    """Choose the default export processing for the detected MIB source."""
    detector_width, detector_height = props.merlin_size
    is_rectangular_eels = detector_width != detector_height

    return {
        "sum_y": is_rectangular_eels,
        "bin_factor": 1,
        "bin_method": "mean",
    }


def export_mib_to_numpy_json(
    input_path: str | Path,
    output_base: str | Path | None = None,
    force: bool = False,
    progress_every: int = 64,
    progress_callback: Optional[ProgressCallback] = None,
    processing_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Export a MIB file to ``.npy`` and ``.json`` outputs.

    Parameters
    ----------
    input_path:
        Path to the source `.mib` file.
    output_base:
        Base output path. If omitted, the input stem is used in the same
        directory as the source file.
    force:
        Overwrite existing outputs if they already exist.
    progress_every:
        Emit progress updates every N frames.
    progress_callback:
        Optional callback receiving ``(percent, message)`` updates.
    """
    input_path = Path(input_path).expanduser().resolve()
    if output_base is None:
        output_base = input_path.with_suffix("")
    output_base = Path(output_base).expanduser().resolve()

    header_fields, file_size = _read_header(input_path)
    props = get_mib_properties(header_fields)

    if props.raw:
        raise ValueError("RAW MIB data is not supported by this exporter.")

    if processing_options is None:
        processing_options = _default_processing_options(props)
    else:
        processing_options = {
            "sum_y": bool(processing_options.get("sum_y", False)),
            "bin_factor": int(processing_options.get("bin_factor", 1) or 1),
            "bin_method": str(processing_options.get("bin_method", "mean")),
        }

    merlin_frame_dtype = np.dtype(
        [
            ("header", np.bytes_, props.headsize),
            ("data", props.pixeltype, props.merlin_size),
        ]
    )

    frame_size_bytes = merlin_frame_dtype.itemsize
    num_frames, remainder = divmod(file_size, frame_size_bytes)
    if remainder:
        print(
            f"Warning: file size is not an exact multiple of frame size; "
            f"ignoring trailing {remainder} bytes."
        )

    scan_size = auto_detect_scan_size(num_frames)
    props.path = str(input_path)
    props.scan_size = scan_size
    props.numberOfFramesInFile = num_frames
    props.xy = scan_size[0] * scan_size[1]

    source = np.memmap(
        input_path,
        dtype=merlin_frame_dtype,
        mode="r",
        offset=props.offset,
        shape=scan_size,
    )

    raw_data = source["data"]
    scan_y, scan_x = raw_data.shape[:2]
    current_dy, current_dx = raw_data.shape[2:4]
    header_dy, header_dx = props.merlin_size

    swap_detector_axes = (
        current_dy != current_dx
        and current_dy > current_dx
        and {current_dy, current_dx} == {header_dy, header_dx}
    )

    if swap_detector_axes:
        canonical_frame_shape = (current_dx, current_dy)
    else:
        canonical_frame_shape = (current_dy, current_dx)

    processed_frame_shape, _ = calculate_processed_size(
        (1, 1, canonical_frame_shape[0], canonical_frame_shape[1]),
        processing_options,
    )
    output_shape = (scan_y, scan_x, processed_frame_shape[2], processed_frame_shape[3])

    output_npy = output_base.with_suffix(".npy")
    output_json = output_base.with_suffix(".json")
    if not force:
        if output_npy.exists():
            raise FileExistsError(f"Output file already exists: {output_npy}")
        if output_json.exists():
            raise FileExistsError(f"Output file already exists: {output_json}")

    output_npy.parent.mkdir(parents=True, exist_ok=True)

    out = np.lib.format.open_memmap(
        output_npy,
        mode="w+",
        dtype=props.pixeltype,
        shape=output_shape,
    )

    total_frames = scan_y * scan_x
    frame_index = 0
    for y in range(scan_y):
        for x in range(scan_x):
            frame = raw_data[y, x]
            if swap_detector_axes:
                frame = np.asarray(frame).reshape(current_dx, current_dy)
            else:
                frame = np.asarray(frame)

            frame_4d = frame[np.newaxis, np.newaxis, :, :]
            frame = apply_data_processing(frame_4d, processing_options)[0, 0]

            out[y, x] = frame

            frame_index += 1
            if progress_callback and (
                frame_index % progress_every == 0 or frame_index == total_frames
            ):
                percent = int((frame_index / total_frames) * 100)
                progress_callback(percent, f"Exported {frame_index:,}/{total_frames:,} frames")

    out.flush()

    metadata = _build_metadata(
        input_path=input_path,
        header_fields=header_fields,
        file_size=file_size,
        props=props,
        num_frames=num_frames,
        scan_size=scan_size,
        output_shape=output_shape,
        output_path=output_npy,
        processing_options=processing_options,
    )
    metadata["swap_detector_axes"] = swap_detector_axes
    metadata["frame_size_bytes"] = frame_size_bytes
    metadata["file_remainder_bytes"] = remainder

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2, ensure_ascii=True)
        f.write("\n")

    return metadata
