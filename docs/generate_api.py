"""Generate the typed Python API document consumed by the Dev Docs reference.

The generator reads the client source with the standard-library ``ast`` module and writes one
deterministic JSON file. It resolves signatures, annotations, decorators, class members, module
constants, docstring sections, and the package's ``__all__`` surface. It does not import the
package, so it runs without the runtime dependencies and cannot execute client code.

Usage:
    uv run --directory extensions/python python docs/generate_api.py \
        --out ../../src/lib/docs/python-api.generated.json

    uv run --directory extensions/python python docs/generate_api.py --check \
        --out ../../src/lib/docs/python-api.generated.json

``--check`` exits non-zero when the committed document differs from a fresh generation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PACKAGE = "xrayatlas"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "src" / "xrayatlas"
DEFAULT_SNIPPETS = Path(__file__).resolve().parent / "snippets"

# Documentation snippets that demonstrate each module. Names map to files in docs/snippets.
MODULE_USAGE: dict[str, list[str]] = {
    "xrayatlas": ["quickstart", "reproduce"],
    "xrayatlas.client": ["explicit_client"],
    "xrayatlas.datasets": ["dataset_reads", "dataset_search"],
    "xrayatlas.models": ["model_reads", "tensor_read"],
    "xrayatlas.publish": ["build_model", "per_channel", "after_publish"],
    "xrayatlas.compare": ["compare_models"],
    "xrayatlas.lock": ["reproduce"],
    "xrayatlas.tensor": ["tensor_read"],
    "xrayatlas.types": ["build_model", "per_channel"],
}

SECTIONS = {
    "args": "parameters",
    "arguments": "parameters",
    "parameters": "parameters",
    "keyword args": "parameters",
    "keyword arguments": "parameters",
    "returns": "returns",
    "return": "returns",
    "yields": "yields",
    "raises": "raises",
    "attributes": "attributes",
    "note": "note",
    "notes": "note",
    "example": "example",
    "examples": "example",
}
_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z ]{1,30}):\s*(.*)$")
_ENTRY_RE = re.compile(r"^(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)(?:\s*\(([^)]*)\))?\s*:\s*(.*)$")
_RETURN_ANNOTATION_RE = re.compile(r"^([A-Za-z_][^\s:]*)\s*:\s*(.*)$")


def main() -> int:
    """Generate or check the committed API document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Package source root")
    parser.add_argument("--snippets", type=Path, default=DEFAULT_SNIPPETS, help="Snippet root")
    parser.add_argument("--out", type=Path, required=True, help="Committed JSON output path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the output file differs from a fresh generation",
    )
    args = parser.parse_args()

    document = build_document(args.source, args.snippets)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        try:
            current = args.out.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"{args.out} is missing; run the generator.", file=sys.stderr)
            return 1
        if current != text:
            print(
                f"{args.out} is out of date; run `bun run docs:python-api`.",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


def build_document(source_root: Path, snippets_root: Path) -> dict[str, Any]:
    """Build the complete API document from a package source tree."""
    files = sorted(path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts)
    modules = [parse_module(path, source_root) for path in files]
    by_name = {module["name"]: module for module in modules}
    exports = resolve_exports(by_name)
    metadata = project_metadata(source_root)
    slim(modules)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "package": PACKAGE,
        "version": metadata["version"],
        "pythonRequires": metadata["pythonRequires"],
        "sourceRoot": f"src/{PACKAGE}",
        "modules": modules,
        "exports": exports,
        "snippets": snippet_files(snippets_root),
    }


def slim(modules: list[dict[str, Any]]) -> None:
    """Drop fields used only while building the document."""
    for module in modules:
        module.pop("imports", None)
        module.pop("docstring", None)
        for item in [*module["classes"], *module["functions"]]:
            item.pop("docstring", None)
            for method in item.get("methods", []):
                method.pop("docstring", None)


def project_metadata(source_root: Path) -> dict[str, str | None]:
    """Read the package version and Python floor from ``pyproject.toml``."""
    try:
        text = (source_root.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": None, "pythonRequires": None}

    def field(name: str) -> str | None:
        match = re.search(rf'^{re.escape(name)}\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else None

    return {"version": field("version"), "pythonRequires": field("requires-python")}


def parse_module(path: Path, source_root: Path) -> dict[str, Any]:
    """Parse one Python module into a JSON-ready description."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    relative = path.relative_to(source_root)
    name = module_name(relative)
    docstring = ast.get_docstring(tree, clean=False) or ""
    module: dict[str, Any] = {
        "name": name,
        "shortName": relative.stem if relative.name != "__init__.py" else name,
        "public": not any(part.startswith("_") for part in relative.parts),
        "source": f"src/xrayatlas/{relative.as_posix()}",
        "sourceSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "line": 1,
        "docstring": docstring,
        "doc": parse_docstring(docstring),
        "classes": [],
        "functions": [],
        "attributes": [],
        "imports": imports_of(tree),
        "usage": MODULE_USAGE.get(name, []),
    }

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                module["classes"].append(parse_class(node, name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                module["functions"].append(parse_function(node, name, owner=None))
        elif isinstance(node, ast.AnnAssign):
            attribute = parse_ann_assign(node, name)
            if attribute is not None:
                module["attributes"].append(attribute)
        elif isinstance(node, ast.Assign):
            attribute = parse_assign(node, name)
            if attribute is not None:
                module["attributes"].append(attribute)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            module["all"] = string_list(node.value)
    sort_members(module)
    return module


def module_name(relative: Path) -> str:
    """Map a package-relative file path to its dotted module name."""
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join([PACKAGE, *parts]) if parts else PACKAGE


def imports_of(tree: ast.Module) -> list[dict[str, Any]]:
    """Return module import bindings used to resolve re-exports."""
    imports: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    {
                        "name": alias.asname or alias.name,
                        "module": node.module,
                        "target": alias.name,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "name": alias.asname or alias.name.split(".")[0],
                        "module": alias.name,
                        "target": alias.name,
                        "line": node.lineno,
                    }
                )
    return imports


def resolve_exports(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve every ``__all__`` name to its defining module and import path."""
    package = by_name.get(PACKAGE)
    if package is None:
        return []
    imports = {item["name"]: item for item in package["imports"]}
    index = member_index(by_name)
    exports: list[dict[str, Any]] = []
    for name in package.get("all", []):
        binding = imports.get(name)
        kind = "unknown"
        module = None
        target = None
        if binding is None:
            module = PACKAGE
            target = f"{PACKAGE}.{name}"
        elif binding["module"] == PACKAGE:
            module = f"{PACKAGE}.{binding['target']}"
            target = module
            kind = "module"
        elif binding["module"] is not None:
            module = binding["module"]
            target = f"{module}.{binding['target']}"
        if kind != "module" and target is not None:
            kind = index.get(target, "unknown")
        exports.append(
            {
                "name": name,
                "importPath": f"{PACKAGE}.{name}",
                "objectPath": target,
                "module": module,
                "kind": kind,
            }
        )
    return exports


def member_index(by_name: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map fully qualified member paths to their source kind."""
    index: dict[str, str] = {}
    for module in by_name.values():
        for item in module["classes"]:
            index[f"{module['name']}.{item['name']}"] = item["kind"]
        for item in module["functions"]:
            index[f"{module['name']}.{item['name']}"] = "function"
        for item in module["attributes"]:
            index[f"{module['name']}.{item['name']}"] = item["kind"]
    return index


def parse_class(node: ast.ClassDef, module: str) -> dict[str, Any]:
    """Parse one class, including its fields, methods, and constructor."""
    bases = [annotation_text(base) for base in node.bases]
    decorators = [annotation_text(item) for item in node.decorator_list]
    fields: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    class_attributes: list[dict[str, Any]] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            field = parse_ann_assign(child, module, owner=node.name)
            if field is not None and field["name"] != "model_config":
                field["default"] = field["value"]
                fields.append(field)
        elif isinstance(child, ast.Assign):
            attribute = parse_assign(child, module, owner=node.name)
            if attribute is not None and attribute["name"] != "model_config":
                class_attributes.append(attribute)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not child.name.startswith("_"):
                methods.append(parse_function(child, module, owner=node.name))
    kind = class_kind(node, bases, decorators)
    docstring = ast.get_docstring(node, clean=False) or ""
    init = next(
        (
            child
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == "__init__"
        ),
        None,
    )
    init_docstring = ast.get_docstring(init, clean=False) if init is not None else None
    return {
        "name": node.name,
        "qualname": f"{module}.{node.name}",
        "module": module,
        "importPath": f"{module}.{node.name}",
        "kind": kind,
        "bases": bases,
        "decorators": decorators,
        "docstring": docstring,
        "doc": parse_docstring(docstring),
        "constructor": parse_docstring(init_docstring) if init_docstring else None,
        "signature": class_signature(node.name, init, fields),
        "parameters": constructor_parameters(init, fields),
        "fields": fields,
        "attributes": class_attributes,
        "methods": methods,
        "line": node.lineno,
        "endLine": getattr(node, "end_lineno", node.lineno),
    }


def class_kind(node: ast.ClassDef, bases: list[str], decorators: list[str]) -> str:
    """Classify a class for the reference badge."""
    names = [base.split(".")[-1] for base in bases]
    if any(name.endswith(("Error", "Exception")) for name in names):
        return "exception"
    if any(name in {"Enum", "IntEnum", "StrEnum"} for name in names):
        return "enum"
    if any(name == "BaseModel" for name in names):
        return "pydantic"
    if any(name == "dataclass" for name in decorators):
        return "dataclass"
    return "class"


def class_signature(
    name: str,
    init: ast.FunctionDef | None,
    fields: list[dict[str, Any]],
) -> str:
    """Build a display signature for a class constructor."""
    if init is not None:
        parameters = function_parameters(init, owner=name)
        return f"{name}({parameter_list_text(parameters)})"
    if fields:
        return f"{name}({parameter_list_text(fields)})"
    return f"{name}()"


def constructor_parameters(
    init: ast.FunctionDef | None,
    fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parameters shown for a class constructor."""
    if init is not None:
        return function_parameters(init, owner="ctor")
    return fields


def parse_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module: str,
    *,
    owner: str | None,
) -> dict[str, Any]:
    """Parse one function or method."""
    docstring = ast.get_docstring(node, clean=False) or ""
    parameters = function_parameters(node, owner=owner)
    return {
        "name": node.name,
        "qualname": f"{module}.{owner + '.' if owner else ''}{node.name}",
        "module": module,
        "importPath": f"{module}.{node.name}" if owner is None else None,
        "kind": "method" if owner else "function",
        "owner": owner,
        "isAsync": isinstance(node, ast.AsyncFunctionDef),
        "isClassMethod": any(
            annotation_text(item) == "classmethod" for item in node.decorator_list
        ),
        "isStaticMethod": any(
            annotation_text(item) == "staticmethod" for item in node.decorator_list
        ),
        "isProperty": any(annotation_text(item) == "property" for item in node.decorator_list),
        "decorators": [annotation_text(item) for item in node.decorator_list],
        "signature": function_signature(node, owner=owner),
        "parameters": parameters,
        "returns": returns_of(node),
        "docstring": docstring,
        "doc": parse_docstring(docstring),
        "line": node.lineno,
        "endLine": getattr(node, "end_lineno", node.lineno),
    }


def function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    owner: str | None,
) -> str:
    """Build the display signature for a function or method."""
    parameters = parameter_list_text(function_parameters(node, owner=owner))
    returns = returns_of(node)
    suffix = f" -> {returns['annotation']}" if returns.get("annotation") else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({parameters}){suffix}"


def function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    owner: str | None,
) -> list[dict[str, Any]]:
    """Return display parameters, dropping ``self`` and ``cls``."""
    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    result: list[dict[str, Any]] = []
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=False)):
        if index == 0 and owner is not None and argument.arg in {"self", "cls"}:
            continue
        result.append(
            parameter(
                argument,
                default,
                kind="positional",
                index=len(result),
            )
        )
    if arguments.vararg is not None:
        result.append(parameter(arguments.vararg, None, kind="vararg", index=len(result)))
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=False):
        result.append(parameter(argument, default, kind="keyword", index=len(result)))
    if arguments.kwarg is not None:
        result.append(parameter(arguments.kwarg, None, kind="kwarg", index=len(result)))
    return result


def parameter(
    argument: ast.arg,
    default: ast.expr | None,
    *,
    kind: str,
    index: int,
) -> dict[str, Any]:
    """Describe one function parameter."""
    annotation = annotation_text(argument.annotation) if argument.annotation else None
    default_text = value_text(default) if default is not None else None
    return {
        "name": argument.arg,
        "annotation": annotation,
        "default": default_text,
        "kind": kind,
        "index": index,
    }


def parameter_list_text(parameters: list[dict[str, Any]]) -> str:
    """Render parameters as Python source, keeping the keyword-only marker."""
    has_vararg = any(item["kind"] == "vararg" for item in parameters)
    parts: list[str] = []
    for item in parameters:
        if item["kind"] == "keyword" and not has_vararg and "*" not in parts:
            parts.append("*")
        parts.append(parameter_text(item))
    return ", ".join(parts)


def parameter_text(item: dict[str, Any]) -> str:
    """Render one parameter as Python source."""
    name = item["name"]
    if item["kind"] == "vararg":
        name = f"*{name}"
    elif item["kind"] == "kwarg":
        name = f"**{name}"
    if item["annotation"]:
        name = f"{name}: {item['annotation']}"
    if item["default"] is not None:
        name = f"{name} = {item['default']}"
    return name


def returns_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Describe a function return annotation."""
    annotation = annotation_text(node.returns) if node.returns is not None else None
    return {"annotation": annotation}


def parse_ann_assign(
    node: ast.AnnAssign,
    module: str,
    *,
    owner: str | None = None,
) -> dict[str, Any] | None:
    """Parse one annotated assignment as a field or constant."""
    if not isinstance(node.target, ast.Name) or node.target.id.startswith("_"):
        return None
    name = node.target.id
    annotation = annotation_text(node.annotation)
    value = value_text(node.value) if node.value is not None else None
    return {
        "name": name,
        "qualname": f"{module}.{owner + '.' if owner else ''}{name}",
        "kind": "typeAlias" if annotation == "TypeAlias" else "attribute",
        "annotation": annotation,
        "value": value,
        "line": node.lineno,
    }


def parse_assign(
    node: ast.Assign,
    module: str,
    *,
    owner: str | None = None,
) -> dict[str, Any] | None:
    """Parse one plain assignment as a constant."""
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    name = node.targets[0].id
    if name.startswith("_") or name == "__all__":
        return None
    value = value_text(node.value)
    return {
        "name": name,
        "qualname": f"{module}.{owner + '.' if owner else ''}{name}",
        "kind": "typeAlias" if type_alias_value(node.value) else "constant",
        "annotation": None,
        "value": value,
        "line": node.lineno,
    }


def type_alias_value(node: ast.expr) -> bool:
    """Heuristic for a module-level type alias written without ``TypeAlias``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return True
    if isinstance(node, ast.Subscript):
        return annotation_text(node.value).split("[")[0] in {
            "list",
            "dict",
            "tuple",
            "set",
            "Sequence",
            "Mapping",
            "Literal",
            "Annotated",
            "Union",
            "Optional",
        }
    return False


def sort_members(module: dict[str, Any]) -> None:
    """Keep source order for members, which is the order the client is read in."""
    module["classes"].sort(key=lambda item: item["line"])
    module["functions"].sort(key=lambda item: item["line"])
    module["attributes"].sort(key=lambda item: item["line"])


def value_text(node: ast.expr | None) -> str:
    """Render an AST expression as source, keeping literal quotes and escapes."""
    if node is None:
        return ""
    return ast.unparse(node)


def annotation_text(node: ast.expr | None) -> str:
    """Render an AST expression as compact source for display."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def string_list(node: ast.expr) -> list[str]:
    """Read a literal list/tuple of strings."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    return [
        item.value
        for item in node.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]


def parse_docstring(docstring: str) -> dict[str, Any]:
    """Split a Google-style docstring into summary, prose, and sections."""
    if not docstring.strip():
        return {"summary": "", "description": "", "sections": []}
    lines = _clean(docstring).splitlines()
    summary: list[str] = []
    description: list[str] = []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            sections.append(current)
            current = None

    for raw in lines:
        stripped = raw.strip()
        match = _SECTION_RE.match(stripped)
        kind = SECTIONS.get(match.group(1).strip().lower()) if match else None
        if kind is not None and match is not None:
            flush()
            current = {
                "name": match.group(1).strip(),
                "kind": kind,
                "lines": [],
                "rawLines": [],
            }
            if match.group(2):
                current["lines"].append(match.group(2))
                current["rawLines"].append(match.group(2))
            continue
        if current is not None:
            current["lines"].append(raw)
            current["rawLines"].append(raw)
        elif not summary:
            if stripped:
                summary.append(stripped)
        elif not stripped and summary and not description:
            continue
        elif stripped:
            description.append(stripped)

    flush()
    for section in sections:
        section["text"] = _wrap(section.pop("lines"))
        section["entries"] = _section_entries(section["kind"], section.pop("rawLines"))
    return {
        "summary": " ".join(summary).strip(),
        "description": " ".join(description).strip(),
        "sections": sections,
    }


def _section_entries(kind: str, lines: list[str]) -> list[dict[str, str]]:
    """Parse ``name: description`` entries from a docstring section."""
    if kind not in {"parameters", "attributes", "raises", "returns", "yields"}:
        return []
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines:
        if not line.strip():
            continue
        match = _ENTRY_RE.match(line.strip())
        return_match = _RETURN_ANNOTATION_RE.match(line.strip()) if kind == "returns" else None
        if match is not None:
            current = {
                "name": match.group(1),
                "type": match.group(2) or "",
                "description": match.group(3).strip(),
            }
            entries.append(current)
        elif return_match is not None and kind == "returns":
            current = {
                "name": return_match.group(1),
                "type": "",
                "description": return_match.group(2).strip(),
            }
            entries.append(current)
        elif current is not None:
            current["description"] = f"{current['description']} {line.strip()}".strip()
    return entries


def _wrap(lines: list[str]) -> str:
    """Dedent section lines and keep paragraph breaks."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in textwrap.dedent("\n".join(lines)).strip("\n").splitlines():
        if line.strip():
            current.append(line.rstrip())
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return "\n\n".join(paragraphs)


def _clean(docstring: str) -> str:
    """Dedent a docstring without turning internal blank lines into spaces."""
    return "\n".join(line.rstrip() for line in docstring.expandtabs().splitlines()).strip("\n")


def snippet_files(root: Path) -> list[dict[str, str]]:
    """List documentation snippets in stable name order."""
    if not root.is_dir():
        return []
    return [{"name": path.stem, "file": path.name} for path in sorted(root.glob("*.py"))]


if __name__ == "__main__":
    raise SystemExit(main())
