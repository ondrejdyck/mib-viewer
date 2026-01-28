from __future__ import annotations

"""Generate AI navigation artifacts for this repo.

Outputs (by default) are written under `docs/ai/`:
  - repo-map.md: human/LLM readable module map
  - symbol-index.json: exact symbol search (functions/classes)

Embeddings are intentionally not generated here.
"""

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Symbol:
    kind: str  # "function" | "class"
    name: str
    qualname: str
    path: str
    lineno: int
    signature: str
    doc: str
    category: str  # "src" | "scripts" | "tests:unit" | ...


def project_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return here


def iter_py_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*.py") if p.is_file()]
    return sorted(files, key=lambda p: str(p).lower())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_line(doc: str) -> str:
    doc = (doc or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def format_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    for a in fn.args.posonlyargs:
        args.append(a.arg)
    if fn.args.posonlyargs:
        args.append("/")
    for a in fn.args.args:
        args.append(a.arg)
    if fn.args.vararg:
        args.append("*" + fn.args.vararg.arg)
    elif fn.args.kwonlyargs:
        args.append("*")
    for a in fn.args.kwonlyargs:
        args.append(a.arg)
    if fn.args.kwarg:
        args.append("**" + fn.args.kwarg.arg)
    return f"{fn.name}({', '.join(args)})"


def classify(rel: str) -> str:
    if rel.startswith("src/"):
        return "src"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("tests/unit/"):
        return "tests:unit"
    if rel.startswith("tests/integration/"):
        return "tests:integration"
    if rel.startswith("tests/"):
        return "tests:other"
    return "other"


def parse_symbols(
    py_path: Path, repo_root: Path, include_private: bool
) -> tuple[str, list[Symbol]]:
    src = read_text(py_path)
    tree = ast.parse(src)
    mod_doc = ast.get_docstring(tree) or ""
    rel = str(py_path.relative_to(repo_root))
    cat = classify(rel)

    out: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not include_private and node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node) or ""
            out.append(
                Symbol(
                    kind="function",
                    name=node.name,
                    qualname=node.name,
                    path=rel,
                    lineno=int(getattr(node, "lineno", 0)),
                    signature=format_signature(node),
                    doc=first_line(doc),
                    category=cat,
                )
            )
        elif isinstance(node, ast.ClassDef):
            if not include_private and node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node) or ""
            out.append(
                Symbol(
                    kind="class",
                    name=node.name,
                    qualname=node.name,
                    path=rel,
                    lineno=int(getattr(node, "lineno", 0)),
                    signature=f"class {node.name}",
                    doc=first_line(doc),
                    category=cat,
                )
            )
    return first_line(mod_doc), out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate docs/ai repo map and symbol index."
    )
    p.add_argument("--out-dir", default="docs/ai", help="Output directory")
    p.add_argument(
        "--include-private", action="store_true", help="Include _private symbols"
    )
    p.add_argument(
        "--include-tests",
        action="store_true",
        help="Include canonical tests (unit + integration) in the symbol index",
    )
    p.add_argument(
        "--include-all-tests",
        action="store_true",
        help="Include all tests/* (including debug/performance/patches) in the symbol index",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    roots: list[Path] = [root / "src", root / "scripts"]
    if args.include_all_tests:
        roots.append(root / "tests")
    elif args.include_tests:
        tests = root / "tests"
        for name in ("unit", "integration"):
            d = tests / name
            if d.exists():
                roots.append(d)

    py_files: list[Path] = []
    for r in roots:
        if r.exists():
            py_files.extend(iter_py_files(r))
    py_files = sorted(set(py_files), key=lambda p: str(p).lower())

    module_list: list[tuple[str, str]] = []
    symbols: list[dict[str, Any]] = []

    for p in py_files:
        doc1, sym = parse_symbols(p, root, include_private=bool(args.include_private))
        rel = str(p.relative_to(root))
        module_list.append((rel, doc1))
        for s in sym:
            symbols.append(
                {
                    "kind": s.kind,
                    "name": s.name,
                    "qualname": s.qualname,
                    "path": s.path,
                    "lineno": s.lineno,
                    "signature": s.signature,
                    "doc": s.doc,
                    "category": s.category,
                }
            )

    symbols = sorted(
        symbols, key=lambda x: (x["name"].lower(), x["path"].lower(), x["lineno"])
    )

    lines: list[str] = []
    lines.append("# Repo map (auto-generated)")
    lines.append("")
    lines.append(
        "This file is generated by `scripts/gen_ai_index.py`. Do not edit manually."
    )
    lines.append("")
    lines.append("## Key locations")
    lines.append("")
    lines.append("- `src/` — application/library code")
    lines.append("- `tests/` — tests (mixed; only some are canonical)")
    lines.append("- `docs/` — documentation")
    lines.append("- `docs/ai/notes/` — high-signal AI session notes")
    lines.append("")
    lines.append("## Modules")
    lines.append("")
    for rel, doc1 in sorted(module_list, key=lambda x: x[0].lower()):
        lines.append(f"- `{rel}`" + (f" — {doc1}" if doc1 else ""))
    lines.append("")
    lines.append("## Symbol index")
    lines.append("")
    lines.append("Generated: `docs/ai/symbol-index.json`")
    lines.append("")
    (out_dir / "repo-map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "symbol-index.json").write_text(
        json.dumps(symbols, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
