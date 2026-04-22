# MIB Viewer notes

This document is the canonical place for the current understanding of the MIB Viewer project.
Keep it short and decision-oriented.

## Goal

Build a stable desktop application for inspecting, converting, and analyzing MIB / EMD / 4D STEM data with a GUI-first workflow.

## Non-goals

- Not building a network service.
- Not optimizing for multi-user or cloud deployment.
- Not introducing new file formats unless needed for concrete workflows.

## Current shape

- `python -m mib_viewer` launches the PyQtGraph-based viewer.
- Core code lives under `src/mib_viewer/`.
- IO and conversion logic lives in `src/mib_viewer/io/`.
- Visualization and app wiring live in `src/mib_viewer/gui/`.
- Processing algorithms live in `src/mib_viewer/processing/`.

## Current workflow sketch

Files → load/inspect → read/convert → visualize → analyze → optional export

Current emphasis:

- MIB loading and conversion remain the core path.
- EMD support is already present for inspection and related workflows.
- 4D STEM and super-resolution features are part of the existing application surface.
- Official MIB export to NumPy/JSON now exists in `src/mib_viewer/io/mib_numpy_exporter.py`
  and is exposed from the GUI as `File -> Export MIB to NumPy/JSON...`.
- AI-facing navigation is kept lightweight through `docs/ai/` artifacts.

## Decisions

- Python 3.10+ is the supported runtime.
- The project is released as a desktop GUI application, not as a backend service.
- AI navigation artifacts should stay small, generated, and easy to refresh.
- Existing notes should be treated as first-class project context.

## Open questions

- Which workflows are stable enough to document as “the way we do it”?
- Should additional development scripts be migrated into a more formal `docs/ai/dev-scripts.md` lifecycle?

## Working conventions

- Keep business logic in `src/mib_viewer/`.
- Keep docs about the AI scaffold in `docs/ai/`.
- Keep session notes in `docs/ai/notes/` with explicit scope and staleness metadata.
