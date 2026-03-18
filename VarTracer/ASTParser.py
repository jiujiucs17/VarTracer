import ast
import json
import os
from collections import defaultdict

DEPENDENCY_GRAPH_COMMENT = {
    "overview": (
        "This JSON is a compact data-dependency graph. Resolve symbol ids through `symbols`, "
        "then use `edges` and `paths` to reconstruct how values flow."
    ),
    "fields": {
        "version": "Schema version for this dependency-graph format. It helps downstream tools detect breaking changes.",
        "trace_started_at": "Timestamp captured when tracing started. It is metadata only and is not part of the dependency graph itself.",
        "files": (
            "Map from short file ids such as `f1` to absolute file paths. "
            "Other fields refer to files through these ids to reduce repetition."
        ),
        "symbols": (
            "Map from short symbol ids such as `s3` to `[kind,name,scope,file_id,line]`. "
            "Each symbol is one node in the graph and usually represents a definition, parameter, or unresolved reference."
        ),
        "symbols.kind": (
            "Category of symbol. Possible values: `var` = ordinary assigned variable; `param` = function parameter; "
            "`func` = function definition symbol; `class` = class definition symbol; `attr` = attribute-like symbol such as "
            "`obj.attr` or an unresolved chained-call name; `subscript` = subscript write target such as `items[]`; "
            "`ref` = external or unresolved reference with no resolved local definition site."
        ),
        "symbols.name": (
            "Raw symbol name captured from tracing and AST dependency analysis. "
            "This is the main human-readable label you should show in explanations."
        ),
        "symbols.scope": (
            "Function-like scope where the symbol is defined or referenced. "
            "`<module>` means module top level."
        ),
        "symbols.file_id": (
            "Short file id pointing back to `files`. "
            "Use it to recover the concrete file path for this symbol."
        ),
        "symbols.line": (
            "Line associated with this symbol. "
            "`0` means no concrete in-file definition line could be resolved."
        ),
        "edges": (
            "List of direct dependency edges stored as `[source_symbol_id,target_symbol_id,edge_kind,line,hits]`. "
            "Each row means the target value directly depends on the source."
        ),
        "edges.source_symbol_id": (
            "Upstream dependency node. "
            "Resolve it through `symbols[source_symbol_id]`."
        ),
        "edges.target_symbol_id": (
            "Downstream node whose value is produced or updated on this step. "
            "Resolve it through `symbols[target_symbol_id]`."
        ),
        "edges.edge_kind": (
            "Type of direct dependency edge. Possible values: `data` = ordinary data dependency from a read value; "
            "`arg` = dependency coming from a function-parameter symbol; `call` = dependency on a function symbol used as a call target; "
            "`ctor` = dependency on a class symbol used as a constructor call."
        ),
        "edges.line": (
            "Line where this direct dependency was observed while building the target symbol. "
            "This is usually the target assignment line."
        ),
        "edges.hits": (
            "Dynamic occurrence count for this exact edge tuple. "
            "Values greater than 1 usually mean the same dependency was seen repeatedly, for example inside loops."
        ),
        "paths": (
            "Map from `sink_symbol_id` to lists of dependency chains. "
            "Each chain is ordered from upstream source to downstream sink and is meant to make end-to-end flow easier to explain."
        ),
        "paths.sink_symbol_id": (
            "Terminal symbol whose value you want to explain. "
            "The corresponding value is a list of paths that all end at this symbol."
        ),
        "paths[*]": (
            "One dependency path represented as a list of symbol ids. "
            "Read it from left to right as source-to-sink flow."
        ),
    },
}

class LineDependencyAnalyzer(ast.NodeVisitor):
    def __init__(self, local_vars, global_vars):
        self.local_vars = set(local_vars)
        self.global_vars = set(global_vars)
        self.dependencies = set()  # 依赖的变量和函数
        self.assigned_vars = set()  # 被赋值的变量
        self.shadowed_scopes = []

    def _is_shadowed(self, name):
        return any(name in scope for scope in reversed(self.shadowed_scopes))

    def _push_shadowed_scope(self, names=None):
        self.shadowed_scopes.append(set(names or []))

    def _pop_shadowed_scope(self):
        if self.shadowed_scopes:
            self.shadowed_scopes.pop()

    def _collect_target_names(self, node):
        names = set()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                names.update(self._collect_target_names(elt))
        return names

    def _maybe_add_name_dependency(self, name):
        if not self._is_shadowed(name) and (name in self.local_vars or name in self.global_vars):
            self.dependencies.add(name)

    def _extract_full_attribute(self, node):
        attr_chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            attr_chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            attr_chain.append(cur.id)
            return '.'.join(reversed(attr_chain)), cur.id
        return None, None

    def _extract_call_root_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            _full_attr, root_name = self._extract_full_attribute(node)
            return root_name
        if isinstance(node, ast.Call):
            return self._extract_call_root_name(node.func)
        return None

    def _extract_called_attribute(self, func_node):
        if not isinstance(func_node, ast.Attribute):
            return None
        root_name = self._extract_call_root_name(func_node.value)
        if root_name and not self._is_shadowed(root_name) and (root_name in self.local_vars or root_name in self.global_vars):
            return f"{root_name}.{func_node.attr}"
        return None

    def _record_load_from_target(self, target):
        if isinstance(target, ast.Name):
            self._maybe_add_name_dependency(target.id)
        elif isinstance(target, ast.Attribute):
            full_attr, root_name = self._extract_full_attribute(target)
            if full_attr and root_name:
                if not self._is_shadowed(root_name) and (root_name in self.local_vars or root_name in self.global_vars):
                    self.dependencies.add(full_attr)
            else:
                self.visit(target.value)
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)
            if isinstance(target.value, ast.Name):
                self._maybe_add_name_dependency(target.value.id)
        else:
            self.visit(target)

    def visit_Name(self, node):
        # 变量名依赖和赋值
        if isinstance(node.ctx, ast.Load):
            self._maybe_add_name_dependency(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.assigned_vars.add(node.id)

    def visit_Attribute(self, node):
        # 递归处理链式属性 obj.a.b.c
        full_attr, root_name = self._extract_full_attribute(node)
        if full_attr and root_name:
            if isinstance(node.ctx, ast.Load):
                if not self._is_shadowed(root_name) and (root_name in self.local_vars or root_name in self.global_vars):
                    self.dependencies.add(full_attr)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                self.assigned_vars.add(full_attr)
        else:
            self.visit(node.value)

    def visit_Call(self, node):
        # 支持链式调用 obj.method1().method2()
        # 新增：无论函数名是否在 local/global，最外层函数名都加入 dependencies
        def add_func_name_to_deps(func_node):
            # 递归找到最左侧的 Name
            if isinstance(func_node, ast.Name):
                self.dependencies.add(func_node.id)
            elif isinstance(func_node, ast.Attribute):
                called_attr = self._extract_called_attribute(func_node)
                if called_attr:
                    self.dependencies.add(called_attr)
                # 递归 Attribute 的 value
                add_func_name_to_deps(func_node.value)
            elif isinstance(func_node, ast.Call):
                add_func_name_to_deps(func_node.func)
        add_func_name_to_deps(node.func)
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Assign(self, node):
        # 支持解包赋值、属性赋值、下标赋值
        for target in node.targets:
            self.visit(target)
        self.visit(node.value)

    def visit_AugAssign(self, node):
        # 处理 x += y
        self._record_load_from_target(node.target)
        self.visit(node.target)
        self.visit(node.value)

    def visit_Subscript(self, node):
        # 处理下标赋值 a[0] = ...
        self.visit(node.value)
        self.visit(node.slice)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            # 赋值目标 a[0]
            if isinstance(node.value, ast.Name):
                self.assigned_vars.add(f"{node.value.id}[]")

    def visit_Lambda(self, node):
        # 处理 lambda 捕获
        arg_names = {arg.arg for arg in node.args.args}
        self._push_shadowed_scope(arg_names)
        try:
            self.visit(node.body)
        finally:
            self._pop_shadowed_scope()

    def visit_ListComp(self, node):
        # 处理列表推导式
        self._push_shadowed_scope()
        try:
            for gen in node.generators:
                self.visit(gen.iter)
                self.shadowed_scopes[-1].update(self._collect_target_names(gen.target))
                for if_clause in gen.ifs:
                    self.visit(if_clause)
            self.visit(node.elt)
        finally:
            self._pop_shadowed_scope()

    def visit_DictComp(self, node):
        self._push_shadowed_scope()
        try:
            for gen in node.generators:
                self.visit(gen.iter)
                self.shadowed_scopes[-1].update(self._collect_target_names(gen.target))
                for if_clause in gen.ifs:
                    self.visit(if_clause)
            self.visit(node.key)
            self.visit(node.value)
        finally:
            self._pop_shadowed_scope()

    def visit_SetComp(self, node):
        self._push_shadowed_scope()
        try:
            for gen in node.generators:
                self.visit(gen.iter)
                self.shadowed_scopes[-1].update(self._collect_target_names(gen.target))
                for if_clause in gen.ifs:
                    self.visit(if_clause)
            self.visit(node.elt)
        finally:
            self._pop_shadowed_scope()

    def visit_GeneratorExp(self, node):
        self._push_shadowed_scope()
        try:
            for gen in node.generators:
                self.visit(gen.iter)
                self.shadowed_scopes[-1].update(self._collect_target_names(gen.target))
                for if_clause in gen.ifs:
                    self.visit(if_clause)
            self.visit(node.elt)
        finally:
            self._pop_shadowed_scope()

    def visit_FunctionDef(self, node):
        # 函数定义在当前作用域绑定函数名；默认参数和装饰器依赖在定义时生效
        self.assigned_vars.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.assigned_vars.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_If(self, node):
        self.visit(node.test)

    def visit_While(self, node):
        self.visit(node.test)

    def visit_For(self, node):
        self.visit(node.target)
        self.visit(node.iter)

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)

    def visit_AsyncWith(self, node):
        self.visit_With(node)

    def visit_Try(self, node):
        return

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        self.visit(node.target)

    def visit_IfExp(self, node):
        # 处理三元表达式
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Tuple(self, node):
        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node):
        for elt in node.elts:
            self.visit(elt)

    def analyze(self, code_line):
        try:
            tree = ast.parse(code_line, mode='exec')
            self.visit(tree)
        except SyntaxError:
            pass
        return {
            "dependencies": self.dependencies,
            "assigned_vars": self.assigned_vars
        }

    def analyze_node(self, node):
        self.visit(node)
        return {
            "dependencies": self.dependencies,
            "assigned_vars": self.assigned_vars
        }
    
class DependencyTree:
    def __init__(self, call_stack=None):
        self.call_stack = None
        self.files = set()
        self.dependency_by_file = None

        if call_stack:
            try:
                self.call_stack = json.loads(call_stack)
                if not isinstance(self.call_stack, dict):
                    raise ValueError("call_stack must be a JSON array.")
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Invalid JSON for call_stack: {e}")
        self.get_files()
        # for item in self.files:
        #     print(item)

    def get_files(self):
        def get_files(call_stack_item):
            if call_stack_item is None:
                return None
            
            files = set()
            details = call_stack_item.get("details", {})

            files.add(details.get("file_path", None))
            daughter_stack = details.get("daughter_stack", None)
            if daughter_stack:
                for item in daughter_stack:
                    files.update(get_files(item))
            return files
        
        if self.call_stack:
            exe_stack = self.call_stack["execution_stack"]
            for item in exe_stack:
                self.files.update(get_files(item))

        return self.files

    def _coerce_line_number(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    def _build_source_index(self):
        source_index = {}

        class DefinitionIndexer(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack = []
                self.params_by_scope = {}
                self.defs_by_scope = {}
                self.defs_by_name = {}

            def _scope_label(self):
                return self.scope_stack[-1] if self.scope_stack else "<module>"

            def _record_def(self, scope, name, kind, line):
                self.defs_by_scope[(scope, name)] = {
                    "kind": kind,
                    "scope": scope,
                    "line": line,
                }
                self.defs_by_name.setdefault(
                    name,
                    {
                        "kind": kind,
                        "scope": scope,
                        "line": line,
                    },
                )

            def _record_params(self, func_name, node):
                params = {}
                positional = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
                for arg in positional:
                    params[arg.arg] = node.lineno
                if node.args.vararg:
                    params[node.args.vararg.arg] = node.lineno
                for arg in node.args.kwonlyargs:
                    params[arg.arg] = node.lineno
                if node.args.kwarg:
                    params[node.args.kwarg.arg] = node.lineno
                self.params_by_scope.setdefault(func_name, {}).update(params)

            def visit_FunctionDef(self, node):
                scope = self._scope_label()
                self._record_def(scope, node.name, "func", node.lineno)
                self._record_params(node.name, node)
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                scope = self._scope_label()
                self._record_def(scope, node.name, "class", node.lineno)
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()

        for file_path in sorted(path for path in self.files if path):
            index = {
                "params_by_scope": {},
                "defs_by_scope": {},
                "defs_by_name": {},
            }
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        tree = ast.parse(handle.read(), filename=file_path)
                    indexer = DefinitionIndexer()
                    indexer.visit(tree)
                    index["params_by_scope"] = indexer.params_by_scope
                    index["defs_by_scope"] = indexer.defs_by_scope
                    index["defs_by_name"] = indexer.defs_by_name
                except (OSError, SyntaxError):
                    pass
            source_index[file_path] = index

        return source_index

    def _infer_assigned_kind(self, name, scope, line_no, file_index):
        if name.endswith("[]"):
            return "subscript"
        if "." in name:
            return "attr"

        definition = file_index.get("defs_by_scope", {}).get((scope, name))
        if definition and definition.get("line") == line_no:
            return definition["kind"]

        return "var"

    def _edge_kind_for_symbol(self, symbol_kind):
        if symbol_kind == "param":
            return "arg"
        if symbol_kind == "func":
            return "call"
        if symbol_kind == "class":
            return "ctor"
        return "data"

    def _build_paths(self, edges, symbols, max_paths_per_sink=8):
        incoming = defaultdict(list)
        outgoing = defaultdict(set)

        for src, dst, _kind, _line, _hits in edges:
            incoming[dst].append(src)
            outgoing[src].add(dst)

        sinks = [
            symbol_id
            for symbol_id in incoming
            if symbol_id not in outgoing and symbols.get(symbol_id, ["var"])[0] not in {"func", "class", "param", "ref"}
        ]

        def walk(symbol_id, visited):
            predecessors = incoming.get(symbol_id, [])
            if not predecessors:
                return [[symbol_id]]

            paths = []
            for predecessor in predecessors:
                if predecessor in visited:
                    continue
                for path in walk(predecessor, visited | {predecessor}):
                    paths.append(path + [symbol_id])
                    if len(paths) >= max_paths_per_sink:
                        return paths
            return paths or [[symbol_id]]

        paths_by_sink = {}
        for sink in sinks:
            sink_paths = walk(sink, {sink})
            if sink_paths:
                paths_by_sink[sink] = sink_paths[:max_paths_per_sink]

        return paths_by_sink
    
    def parse_dependency(self):
        """
        Parse the execution stack into a compact dependency graph that is
        optimized for downstream LLM consumption.
        """
        file_paths = sorted(path for path in self.files if path)
        file_ids = {path: f"f{index}" for index, path in enumerate(file_paths, 1)}
        source_index = self._build_source_index()

        graph = {
            "_comment": DEPENDENCY_GRAPH_COMMENT,
            "version": "ddg.v2",
            "trace_started_at": self.call_stack.get("trace_started_at"),
            "files": {file_id: path for path, file_id in file_ids.items()},
            "symbols": {},
            "edges": [],
            "paths": {},
        }

        symbol_by_key = {}
        latest_symbol = {}
        edge_hits = defaultdict(int)
        next_symbol_id = 1

        def ensure_symbol(kind, name, scope, file_id, line_no):
            nonlocal next_symbol_id
            key = (kind, name, scope, file_id, line_no)
            if key in symbol_by_key:
                return symbol_by_key[key]

            symbol_id = f"s{next_symbol_id}"
            next_symbol_id += 1
            graph["symbols"][symbol_id] = [kind, name, scope, file_id, line_no]
            symbol_by_key[key] = symbol_id
            return symbol_id

        def resolve_dependency_symbol(dep_name, scope, file_path, file_id, line_no):
            same_scope_key = (file_id, scope, dep_name)
            module_scope_key = (file_id, "<module>", dep_name)
            if same_scope_key in latest_symbol:
                return latest_symbol[same_scope_key]
            if module_scope_key in latest_symbol:
                return latest_symbol[module_scope_key]

            file_index = source_index.get(file_path, {})
            param_line = file_index.get("params_by_scope", {}).get(scope, {}).get(dep_name)
            if param_line is not None:
                symbol_id = ensure_symbol("param", dep_name, scope, file_id, param_line)
                latest_symbol[same_scope_key] = symbol_id
                return symbol_id

            definition = file_index.get("defs_by_name", {}).get(dep_name)
            if definition:
                symbol_id = ensure_symbol(
                    definition["kind"],
                    dep_name,
                    definition["scope"],
                    file_id,
                    definition["line"],
                )
                latest_symbol[(file_id, definition["scope"], dep_name)] = symbol_id
                return symbol_id

            kind = "attr" if "." in dep_name else "ref"
            return ensure_symbol(kind, dep_name, scope, file_id, 0)

        def traverse(stack):
            for item in stack:
                details = item.get("details", {})
                event_type = item.get("type")
                file_path = details.get("file_path")
                if file_path not in file_ids:
                    if "daughter_stack" in details:
                        traverse(details["daughter_stack"])
                    continue

                if event_type == "LINE":
                    file_id = file_ids[file_path]
                    scope = details.get("func") or "<module>"
                    line_no = self._coerce_line_number(details.get("line_no"))
                    assigned_vars = details.get("assigned_vars", [])
                    dependencies = details.get("dependencies", [])
                    file_index = source_index.get(file_path, {})

                    target_symbols = {}
                    for var_name in assigned_vars:
                        symbol_kind = self._infer_assigned_kind(var_name, scope, line_no, file_index)
                        target_symbols[var_name] = ensure_symbol(symbol_kind, var_name, scope, file_id, line_no)

                    for var_name, target_id in target_symbols.items():
                        for dep_name in dependencies:
                            source_id = resolve_dependency_symbol(dep_name, scope, file_path, file_id, line_no)
                            source_kind = graph["symbols"][source_id][0]
                            edge_key = (
                                source_id,
                                target_id,
                                self._edge_kind_for_symbol(source_kind),
                                line_no,
                            )
                            edge_hits[edge_key] += 1

                    for var_name, target_id in target_symbols.items():
                        latest_symbol[(file_id, scope, var_name)] = target_id

                if "daughter_stack" in details:
                    traverse(details["daughter_stack"])

        traverse(self.call_stack.get("execution_stack", []))

        graph["edges"] = [
            [src, dst, kind, line_no, hits]
            for (src, dst, kind, line_no), hits in sorted(
                edge_hits.items(),
                key=lambda item: (item[0][3], item[0][0], item[0][1], item[0][2]),
            )
        ]
        graph["paths"] = self._build_paths(graph["edges"], graph["symbols"])

        self.dependency_by_file = graph
        return graph
