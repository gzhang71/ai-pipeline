"""Build a code graph for a Python repository using the standard library `ast`.

The graph has three node kinds beyond the module itself:

    module   -- one per .py file, identified by its dotted import path
    class    -- a `class` statement
    function -- a module-level or nested `def` / `async def`
    method   -- a `def` whose immediate parent is a class

and three edge kinds:

    contains -- module -> class -> method, module -> function -> nested function
    imports  -- module -> module, only when the target resolves to a file in
                the walked tree (external imports are recorded separately)
    calls    -- caller node -> callee node, only when statically resolvable

WHAT THIS ANALYSIS CANNOT DO
----------------------------
`ast` sees syntax, not values. The call graph is therefore a *lower bound* --
every edge it reports is real, but many real edges are missing:

* dynamic imports (`importlib.import_module(name)`, `__import__`) are invisible;
* `getattr(obj, name)()` is invisible;
* attribute calls on values whose type is unknown (`x.run()` where `x` came from
  a parameter, a container, or a function return) are unresolved -- we only
  resolve `self.m()` inside a class, `module.f()` through an import alias, and
  `Class.m()` / bare `f()` through module scope;
* inheritance is not followed: `self.m()` resolves only if `m` is defined on the
  *same* class, not on a base class;
* decorators that replace a function, metaclasses, monkey-patching, `functools`
  wrappers, and conditional definitions all change runtime behaviour invisibly;
* calls through variables bound to functions (`f = slugify; f()`) are unresolved.

`CodeGraph.stats()["unresolved_calls"]` reports how many call sites were seen
but not resolved, so the size of the blind spot is always visible.
"""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

NODE_KINDS = ("module", "class", "function", "method")
EDGE_KINDS = ("contains", "imports", "calls")


# --------------------------------------------------------------------------
# Serializable records produced by the per-file pass
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One addressable symbol. `id` is what the retrieval tools speak in."""

    id: str
    kind: str
    name: str
    qualname: str
    module: str
    path: str
    lineno: int
    end_lineno: int
    signature: str
    doc: str
    parent: str | None
    # Module-level assignment targets (constants, aliases). These are NOT nodes
    # of their own -- a constant has no body to fetch, it lives in its module.
    # Recording the names here is what makes `search_symbols("RETRY_LIMIT")`
    # return the module that defines it, so the model knows which file to read.
    defines: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        data = dict(data)
        data["defines"] = tuple(data.get("defines", ()))
        return cls(**data)

    def outline(self) -> str:
        """One-line rendering: identifier + signature + docstring head. No body."""
        head = f"{self.id}  {self.signature}"
        if self.doc:
            head = f"{head}  -- {self.doc}"
        return head


@dataclass(frozen=True)
class ImportRef:
    kind: str  # "import" | "from"
    module: str | None  # dotted target module, after relative resolution
    name: str | None  # imported attribute for "from" imports
    alias: str  # the local binding used in this module
    lineno: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImportRef":
        return cls(**data)


@dataclass(frozen=True)
class CallRef:
    caller: str  # node id of the enclosing function/method/module
    parts: tuple[str, ...]  # dotted parts of the called expression
    lineno: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parts"] = list(self.parts)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallRef":
        return cls(
            caller=data["caller"],
            parts=tuple(data["parts"]),
            lineno=data["lineno"],
        )


@dataclass
class FileAnalysis:
    """Everything extracted from one file, before cross-file resolution."""

    path: str  # repo-relative, posix separators
    module: str
    mtime: float
    size: int
    sha256: str
    nodes: list[Node] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    calls: list[CallRef] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "module": self.module,
            "mtime": self.mtime,
            "size": self.size,
            "sha256": self.sha256,
            "nodes": [n.to_dict() for n in self.nodes],
            "imports": [i.to_dict() for i in self.imports],
            "calls": [c.to_dict() for c in self.calls],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileAnalysis":
        return cls(
            path=data["path"],
            module=data["module"],
            mtime=data["mtime"],
            size=data["size"],
            sha256=data["sha256"],
            nodes=[Node.from_dict(n) for n in data["nodes"]],
            imports=[ImportRef.from_dict(i) for i in data["imports"]],
            calls=[CallRef.from_dict(c) for c in data["calls"]],
            error=data.get("error"),
        )


# --------------------------------------------------------------------------
# Per-file AST pass
# --------------------------------------------------------------------------


def module_name_for(rel_path: str) -> str:
    """`common/client.py` -> `common.client`, `pkg/__init__.py` -> `pkg`."""
    parts = rel_path.replace(os.sep, "/").split("/")
    parts[-1] = parts[-1][: -len(".py")] if parts[-1].endswith(".py") else parts[-1]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(p for p in parts if p) or "__root__"


def _docline(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)  # type: ignore[arg-type]
    except TypeError:
        return ""
    if not doc:
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _signature(node: ast.AST, name: str) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
        except Exception:  # pragma: no cover - defensive
            args = ", ".join(a.arg for a in node.args.args)
        returns = ""
        if node.returns is not None:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:  # pragma: no cover - defensive
                returns = ""
        return f"{prefix} {name}({args}){returns}"
    if isinstance(node, ast.ClassDef):
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:  # pragma: no cover - defensive
                continue
        return f"class {name}({', '.join(bases)})" if bases else f"class {name}"
    return f"module {name}"


def _start_line(node: ast.AST) -> int:
    lines = [getattr(node, "lineno", 1)]
    for dec in getattr(node, "decorator_list", []) or []:
        lines.append(dec.lineno)
    return min(lines)


def _dotted_parts(node: ast.AST) -> tuple[str, ...] | None:
    """Render `a.b.c` as ('a','b','c'); return None for anything else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return tuple(reversed(parts))
    return None


def _resolve_relative(module: str, level: int, target: str | None) -> str | None:
    """Resolve `from ..pkg import x` against the importing module's package."""
    if level == 0:
        return target
    pkg_parts = module.split(".")
    # a module `a.b.c` sits in package `a.b`; level 1 means that package.
    base = pkg_parts[:-1]
    drop = level - 1
    if drop:
        if drop > len(base):
            return None
        base = base[: len(base) - drop]
    if target:
        base = base + target.split(".")
    return ".".join(base) if base else None


class _ModuleVisitor(ast.NodeVisitor):
    def __init__(self, module: str, path: str) -> None:
        self.module = module
        self.path = path
        self.nodes: list[Node] = []
        self.imports: list[ImportRef] = []
        self.calls: list[CallRef] = []
        self._scope: list[str] = []  # qualname components
        self._kinds: list[str] = []  # kind of each enclosing scope
        self._owner: list[str] = [module]  # node id of the current owner
        self.defines: list[str] = []  # module-level assignment targets

    # -- helpers ---------------------------------------------------------
    def _qualname(self, name: str) -> str:
        return ".".join(self._scope + [name])

    def _add(self, node: ast.AST, name: str, kind: str) -> str:
        qualname = self._qualname(name)
        node_id = f"{self.module}:{qualname}"
        self.nodes.append(
            Node(
                id=node_id,
                kind=kind,
                name=name,
                qualname=qualname,
                module=self.module,
                path=self.path,
                lineno=_start_line(node),
                end_lineno=getattr(node, "end_lineno", _start_line(node)),
                signature=_signature(node, name),
                doc=_docline(node),
                parent=self._owner[-1],
            )
        )
        return node_id

    def _descend(self, node: ast.AST, node_id: str, name: str, kind: str) -> None:
        self._scope.append(name)
        self._kinds.append(kind)
        self._owner.append(node_id)
        self.generic_visit(node)
        self._owner.pop()
        self._kinds.pop()
        self._scope.pop()

    # -- visitors --------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRef(
                    kind="import",
                    module=alias.name,
                    name=None,
                    alias=alias.asname or alias.name,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = _resolve_relative(self.module, node.level or 0, node.module)
        for alias in node.names:
            if alias.name == "*":
                continue
            self.imports.append(
                ImportRef(
                    kind="from",
                    module=target,
                    name=alias.name,
                    alias=alias.asname or alias.name,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def _record_define(self, target: ast.AST) -> None:
        if self._scope:  # module scope only
            return
        if isinstance(target, ast.Name):
            self.defines.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._record_define(element)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_define(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_define(node.target)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        node_id = self._add(node, node.name, "class")
        self._descend(node, node_id, node.name, "class")

    def _visit_func(self, node: ast.AST, name: str) -> None:
        kind = "method" if self._kinds and self._kinds[-1] == "class" else "function"
        node_id = self._add(node, name, kind)
        self._descend(node, node_id, name, kind)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        parts = _dotted_parts(node.func)
        if parts:
            self.calls.append(
                CallRef(caller=self._owner[-1], parts=parts, lineno=node.lineno)
            )
        self.generic_visit(node)


def analyze_source(source: str, *, rel_path: str, module: str) -> tuple[
    list[Node], list[ImportRef], list[CallRef], str | None
]:
    """Parse one module's source. Returns (nodes, imports, calls, error)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], [], [], f"SyntaxError: {exc}"
    line_count = source.count("\n") + 1
    visitor = _ModuleVisitor(module, rel_path)
    visitor.visit(tree)
    module_node = Node(
        id=module,
        kind="module",
        name=module.rsplit(".", 1)[-1],
        qualname=module,
        module=module,
        path=rel_path,
        lineno=1,
        end_lineno=line_count,
        signature=f"module {module}",
        doc=_docline(tree),
        parent=None,
        defines=tuple(dict.fromkeys(visitor.defines)),
    )
    return [module_node, *visitor.nodes], visitor.imports, visitor.calls, None


def analyze_file(abs_path: str, *, root: str, module: str | None = None) -> FileAnalysis:
    rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
    module = module or module_name_for(rel_path)
    raw = open(abs_path, "rb").read()
    stat = os.stat(abs_path)
    source = raw.decode("utf-8", errors="replace")
    nodes, imports, calls, error = analyze_source(
        source, rel_path=rel_path, module=module
    )
    return FileAnalysis(
        path=rel_path,
        module=module,
        mtime=stat.st_mtime,
        size=stat.st_size,
        sha256=hashlib.sha256(raw).hexdigest(),
        nodes=nodes,
        imports=imports,
        calls=calls,
        error=error,
    )


# --------------------------------------------------------------------------
# Cross-file resolution
# --------------------------------------------------------------------------


class CodeGraph:
    """Resolved graph over a set of `FileAnalysis` records."""

    def __init__(self, root: str, files: dict[str, FileAnalysis]) -> None:
        self.root = os.path.abspath(root)
        self.files = files
        self.nodes: dict[str, Node] = {}
        self.modules: dict[str, str] = {}  # module name -> path
        self.children: dict[str, list[str]] = {}
        self.imports: dict[str, list[str]] = {}
        self.imported_by: dict[str, list[str]] = {}
        self.external_imports: dict[str, list[str]] = {}
        self.callees: dict[str, list[str]] = {}
        self.callers: dict[str, list[str]] = {}
        self.unresolved_calls: list[CallRef] = []
        self._resolve()

    # -- construction ----------------------------------------------------
    def _resolve(self) -> None:
        for analysis in self.files.values():
            for node in analysis.nodes:
                self.nodes[node.id] = node
            self.modules.setdefault(analysis.module, analysis.path)

        for node in self.nodes.values():
            if node.parent:
                self.children.setdefault(node.parent, []).append(node.id)
        for kids in self.children.values():
            kids.sort()

        aliases: dict[str, dict[str, tuple[str, str]]] = {}
        for analysis in self.files.values():
            table: dict[str, tuple[str, str]] = {}
            for ref in analysis.imports:
                if not ref.module:
                    continue
                if ref.kind == "import":
                    if ref.module in self.modules:
                        table[ref.alias] = ("module", ref.module)
                        self._add_import(analysis.module, ref.module)
                    else:
                        self.external_imports.setdefault(analysis.module, []).append(
                            ref.module
                        )
                    continue
                # from X import Y -- Y may be a submodule or a symbol
                submodule = f"{ref.module}.{ref.name}"
                if submodule in self.modules:
                    table[ref.alias] = ("module", submodule)
                    self._add_import(analysis.module, submodule)
                elif ref.module in self.modules:
                    symbol_id = f"{ref.module}:{ref.name}"
                    self._add_import(analysis.module, ref.module)
                    if symbol_id in self.nodes:
                        table[ref.alias] = ("node", symbol_id)
                    else:
                        table[ref.alias] = ("module", ref.module)
                else:
                    self.external_imports.setdefault(analysis.module, []).append(
                        f"{ref.module}.{ref.name}"
                    )
            aliases[analysis.module] = table

        for analysis in self.files.values():
            table = aliases.get(analysis.module, {})
            for call in analysis.calls:
                target = self._resolve_call(analysis.module, call, table)
                if target is None:
                    self.unresolved_calls.append(call)
                    continue
                self.callees.setdefault(call.caller, [])
                if target not in self.callees[call.caller]:
                    self.callees[call.caller].append(target)
                self.callers.setdefault(target, [])
                if call.caller not in self.callers[target]:
                    self.callers[target].append(call.caller)

    def _add_import(self, source: str, target: str) -> None:
        if source == target:
            return
        edges = self.imports.setdefault(source, [])
        if target not in edges:
            edges.append(target)
        back = self.imported_by.setdefault(target, [])
        if source not in back:
            back.append(source)

    def _enclosing_class(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        while node is not None and node.parent:
            parent = self.nodes.get(node.parent)
            if parent is None:
                return None
            if parent.kind == "class":
                return parent.id
            node = parent
        return None

    def _resolve_call(
        self, module: str, call: CallRef, table: dict[str, tuple[str, str]]
    ) -> str | None:
        parts = call.parts

        # self.method() inside a class
        if parts[0] in ("self", "cls") and len(parts) == 2:
            cls = self._enclosing_class(call.caller)
            if cls:
                candidate = f"{cls}.{parts[1]}"
                if candidate in self.nodes:
                    return candidate
            return None

        # longest import-alias prefix wins
        for cut in range(len(parts), 0, -1):
            alias = ".".join(parts[:cut])
            entry = table.get(alias)
            if entry is None:
                continue
            kind, target = entry
            rest = parts[cut:]
            if not rest:
                return target if target in self.nodes else None
            if kind == "module":
                candidate = f"{target}:{'.'.join(rest)}"
            else:
                candidate = f"{target}.{'.'.join(rest)}"
            return candidate if candidate in self.nodes else None

        # bare name: walk out through the enclosing scopes of the caller
        if len(parts) == 1:
            caller = self.nodes.get(call.caller)
            scope = caller.qualname if caller and caller.kind != "module" else ""
            prefix_parts = scope.split(".") if scope else []
            while True:
                qual = ".".join([*prefix_parts, parts[0]]) if prefix_parts else parts[0]
                candidate = f"{module}:{qual}"
                if candidate in self.nodes:
                    return candidate
                if not prefix_parts:
                    return None
                prefix_parts.pop()

        # Class.method() defined in this module
        candidate = f"{module}:{'.'.join(parts)}"
        return candidate if candidate in self.nodes else None

    # -- accessors -------------------------------------------------------
    def node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def iter_nodes(self, kinds: Iterable[str] | None = None) -> list[Node]:
        wanted = set(kinds) if kinds else None
        out = [n for n in self.nodes.values() if wanted is None or n.kind in wanted]
        out.sort(key=lambda n: (n.path, n.lineno, n.id))
        return out

    def source_of(self, node: Node) -> str:
        abs_path = os.path.join(self.root, node.path)
        with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    def stats(self) -> dict[str, int]:
        kinds = {kind: 0 for kind in NODE_KINDS}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        return {
            "files": len(self.files),
            **{f"{k}s": v for k, v in kinds.items()},
            "contains_edges": sum(len(v) for v in self.children.values()),
            "import_edges": sum(len(v) for v in self.imports.values()),
            "call_edges": sum(len(v) for v in self.callees.values()),
            "unresolved_calls": len(self.unresolved_calls),
            "external_imports": sum(len(v) for v in self.external_imports.values()),
        }
