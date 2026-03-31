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
            "`ctor` = dependency on a class symbol used as a constructor call; `ret` = dependency carried back from a traced callee return."
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

FLOW_TRACE_COMMENT = {
    "overview": (
        "This JSON is a compact runtime flow trace. Use `frames` to recover the dynamic call tree, "
        "`symbols` to resolve ids, and `steps` to read how values and control moved during execution."
    ),
    "fields": {
        "version": "Schema version for this runtime flow-trace format.",
        "trace_started_at": "Timestamp captured when tracing started. It is metadata only.",
        "files": (
            "Map from short file ids such as `f1` to absolute file paths. "
            "Other sections refer to files through these ids."
        ),
        "symbols": (
            "Map from short symbol ids such as `s3` to `[kind,name,scope,file_id,line]`. "
            "The layout matches the dependency graph so the same symbol ids can be reused across both artifacts."
        ),
        "frames": (
            "Map from dynamic frame ids such as `c2` to `[parent_frame_id,func,file_id,depth,call_line]`. "
            "An empty parent id means the frame was entered from the trace root rather than from another traced frame."
        ),
        "frames.parent_frame_id": "Dynamic caller frame id, or an empty string for root-entered frames.",
        "frames.func": "Function-like name for this runtime frame, such as `<module>`, `main`, or `run`.",
        "frames.file_id": "Short file id pointing back to `files`.",
        "frames.depth": "Call depth copied from the execution trace.",
        "frames.call_line": (
            "Caller-side line that triggered this frame. "
            "`0` means the entry was not matched back to a concrete traced call site."
        ),
        "steps": (
            "Ordered list of runtime flow steps. Each step is stored as "
            "`[seq,frame_id,op,line,reads,writes,meta]`."
        ),
        "steps.seq": "Global execution-order counter starting at 1.",
        "steps.frame_id": "Dynamic frame id owning this step. It may be an empty string for loose root-level steps.",
        "steps.op": (
            "Operation kind. Values: `def` = definition binding; `as` = direct assignment/update; "
            "`use` = read-only statement; `cond` = control-flow test; `call` = caller enters a traced callee; "
            "`arg` = actual-to-parameter transfer; `bind` = callee result bound back into caller state; "
            "`ret` = current frame returns data outward; `exc` = exception observed on this line."
        ),
        "steps.line": "Concrete source line for this runtime step, or `0` when unavailable.",
        "steps.reads": "List of upstream symbol ids consumed by this step.",
        "steps.writes": "List of downstream symbol ids produced or updated by this step.",
        "steps.meta": (
            "Small metadata dictionary used only when needed. "
            "Typical keys are `stmt` for the AST statement kind, `callee` for a child frame id, "
            "`func` for a callee name, or `exception_type` for exception steps."
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


class ExpressionDependencyCollector(ast.NodeVisitor):
    """Collect potential dependency labels from an expression without runtime symbol filtering."""

    def __init__(self):
        self.dependencies = set()

    def _extract_full_attribute(self, node):
        attr_chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            attr_chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            attr_chain.append(cur.id)
            return ".".join(reversed(attr_chain)), cur.id
        return None, None

    def _extract_called_attribute(self, func_node):
        if not isinstance(func_node, ast.Attribute):
            return None
        full_attr, _root_name = self._extract_full_attribute(func_node)
        return full_attr

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.dependencies.add(node.id)

    def visit_Attribute(self, node):
        full_attr, _root_name = self._extract_full_attribute(node)
        if full_attr and isinstance(node.ctx, ast.Load):
            self.dependencies.add(full_attr)
        self.visit(node.value)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.dependencies.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_attr = self._extract_called_attribute(node.func)
            if called_attr:
                self.dependencies.add(called_attr)
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def collect(self, node):
        self.visit(node)
        return self.dependencies


class RuntimeStatementAnalyzer:
    """Summarize runtime-relevant flow semantics for one executed statement."""

    def __init__(self, dependencies, assigned_vars):
        self.dependencies = set(dependencies or [])
        self.assigned_vars = list(assigned_vars or [])

    def _collect_expr_dependencies(self, node):
        if node is None:
            return []
        deps = ExpressionDependencyCollector().collect(node)
        return sorted(dep for dep in deps if dep in self.dependencies)

    def _collect_call_nodes(self, node):
        calls = []

        class CallCollector(ast.NodeVisitor):
            def visit_Call(self, call_node):
                self.visit(call_node.func)
                for arg in call_node.args:
                    self.visit(arg)
                for kw in call_node.keywords:
                    self.visit(kw.value)
                calls.append(call_node)

        CallCollector().visit(node)
        return calls

    def _callee_label(self, func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id, func_node.id
        if isinstance(func_node, ast.Attribute):
            full_attr, _root_name = ExpressionDependencyCollector()._extract_full_attribute(func_node)
            if full_attr:
                return full_attr, func_node.attr
            return func_node.attr, func_node.attr
        return None, None

    def analyze(self, stmt_node):
        stmt_kind = type(stmt_node).__name__
        if isinstance(stmt_node, ast.Return):
            result_kind = "return"
            result_targets = ["@return"]
        elif isinstance(stmt_node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result_kind = "assign"
            result_targets = list(self.assigned_vars)
        else:
            result_kind = "none"
            result_targets = []

        callsites = []
        consumed_dependencies = set()
        for call_node in self._collect_call_nodes(stmt_node):
            callee_label, callee_hint = self._callee_label(call_node.func)
            receiver_groups = []
            if isinstance(call_node.func, ast.Attribute):
                receiver_groups.append(self._collect_expr_dependencies(call_node.func.value))

            arg_groups = receiver_groups
            for arg in call_node.args:
                arg_groups.append(self._collect_expr_dependencies(arg))
            for kw in call_node.keywords:
                arg_groups.append(self._collect_expr_dependencies(kw.value))

            call_deps = set(self._collect_expr_dependencies(call_node))
            callable_deps = self._collect_expr_dependencies(call_node.func)
            consumed_dependencies.update(call_deps)
            callsites.append(
                {
                    "callee_label": callee_label,
                    "callee_hint": callee_hint,
                    "callable_dependencies": callable_deps,
                    "consumed_dependencies": sorted(call_deps),
                    "arg_dependency_groups": arg_groups,
                    "result_kind": result_kind,
                    "result_targets": list(result_targets),
                }
            )

        residual_dependencies = sorted(dep for dep in self.dependencies if dep not in consumed_dependencies)
        return {
            "stmt_kind": stmt_kind,
            "result_kind": result_kind,
            "result_targets": result_targets,
            "callsites": callsites,
            "residual_dependencies": residual_dependencies,
        }
    
class DependencyTree:
    def __init__(self, call_stack=None):
        self.call_stack = None
        self.files = set()
        self.dependency_by_file = None
        self.flow_trace = None
        self._source_index_cache = None
        self._artifact_cache = None

        if call_stack:
            if isinstance(call_stack, dict):
                self.call_stack = call_stack
            else:
                try:
                    self.call_stack = json.loads(call_stack)
                    if not isinstance(self.call_stack, dict):
                        raise ValueError("call_stack must be a JSON object.")
                except (TypeError, json.JSONDecodeError, ValueError) as e:
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
        if self._source_index_cache is not None:
            return self._source_index_cache

        source_index = {}

        class DefinitionIndexer(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack = []
                self.params_by_scope = {}
                self.param_order_by_scope = {}
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
                order = []
                positional = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
                for arg in positional:
                    params[arg.arg] = node.lineno
                    order.append(arg.arg)
                if node.args.vararg:
                    params[node.args.vararg.arg] = node.lineno
                    order.append(node.args.vararg.arg)
                for arg in node.args.kwonlyargs:
                    params[arg.arg] = node.lineno
                    order.append(arg.arg)
                if node.args.kwarg:
                    params[node.args.kwarg.arg] = node.lineno
                    order.append(node.args.kwarg.arg)
                self.params_by_scope.setdefault(func_name, {}).update(params)
                self.param_order_by_scope.setdefault(func_name, [])
                for name in order:
                    if name not in self.param_order_by_scope[func_name]:
                        self.param_order_by_scope[func_name].append(name)

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
                "param_order_by_scope": {},
                "defs_by_scope": {},
                "defs_by_name": {},
                "stmt_by_line": {},
            }
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        tree = ast.parse(handle.read(), filename=file_path)
                    indexer = DefinitionIndexer()
                    indexer.visit(tree)
                    index["params_by_scope"] = indexer.params_by_scope
                    index["param_order_by_scope"] = indexer.param_order_by_scope
                    index["defs_by_scope"] = indexer.defs_by_scope
                    index["defs_by_name"] = indexer.defs_by_name
                    stmt_nodes = [
                        node for node in ast.walk(tree)
                        if isinstance(node, ast.stmt) and hasattr(node, "lineno") and hasattr(node, "end_lineno")
                    ]
                    line_to_stmt = {}
                    for node in stmt_nodes:
                        key = self._statement_key(node)
                        for line_no in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                            current = line_to_stmt.get(line_no)
                            if current is None or key < current[0]:
                                line_to_stmt[line_no] = (key, node)
                    index["stmt_by_line"] = {
                        line_no: cached_node
                        for line_no, (_key, cached_node) in line_to_stmt.items()
                    }
                except (OSError, SyntaxError):
                    pass
            source_index[file_path] = index

        self._source_index_cache = source_index
        return source_index

    def _statement_key(self, node):
        end_lineno = getattr(node, "end_lineno", node.lineno)
        end_col = getattr(node, "end_col_offset", 0)
        start_col = getattr(node, "col_offset", 0)
        return (end_lineno - node.lineno, end_col - start_col)

    def _get_statement_node(self, file_index, line_no):
        return file_index.get("stmt_by_line", {}).get(line_no)

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
    
    def _build_semantic_artifacts(self):
        if self._artifact_cache is not None:
            return self._artifact_cache

        file_paths = sorted(path for path in self.files if path)
        file_ids = {path: f"f{index}" for index, path in enumerate(file_paths, 1)}
        source_index = self._build_source_index()

        graph = {
            "_comment": DEPENDENCY_GRAPH_COMMENT,
            "version": "ddg.v3",
            "trace_started_at": self.call_stack.get("trace_started_at"),
            "files": {file_id: path for path, file_id in file_ids.items()},
            "symbols": {},
            "edges": [],
            "paths": {},
        }
        flow = {
            "_comment": FLOW_TRACE_COMMENT,
            "version": "flow.v1",
            "trace_started_at": self.call_stack.get("trace_started_at"),
            "files": graph["files"],
            "symbols": graph["symbols"],
            "frames": {},
            "steps": [],
        }

        symbol_by_key = {}
        latest_symbol = {}
        edge_hits = defaultdict(int)
        statement_analysis_cache = {}
        target_symbol_cache = {}
        next_symbol_id = 1
        next_frame_id = 1
        next_step_seq = 1

        def uniq(values):
            seen = set()
            ordered = []
            for value in values:
                if value and value not in seen:
                    seen.add(value)
                    ordered.append(value)
            return ordered

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

        def resolve_many(dep_names, scope, file_path, file_id, line_no):
            return uniq(
                resolve_dependency_symbol(dep_name, scope, file_path, file_id, line_no)
                for dep_name in dep_names or []
            )

        def add_edge(source_id, target_id, line_no, kind=None):
            source_kind = graph["symbols"][source_id][0]
            edge_kind = kind or self._edge_kind_for_symbol(source_kind)
            edge_hits[(source_id, target_id, edge_kind, line_no)] += 1

        def add_step(frame_id, op, line_no, reads=None, writes=None, meta=None):
            nonlocal next_step_seq
            flow["steps"].append(
                [
                    next_step_seq,
                    frame_id or "",
                    op,
                    line_no,
                    uniq(reads or []),
                    uniq(writes or []),
                    meta or {},
                ]
            )
            next_step_seq += 1

        def ensure_targets(target_names, scope, file_id, line_no, file_index):
            cache_key = (file_id, scope, line_no, tuple(target_names))
            cached_targets = target_symbol_cache.get(cache_key)
            if cached_targets is not None:
                return cached_targets

            targets = {}
            for var_name in target_names:
                symbol_kind = self._infer_assigned_kind(var_name, scope, line_no, file_index)
                targets[var_name] = ensure_symbol(symbol_kind, var_name, scope, file_id, line_no)

            target_symbol_cache[cache_key] = targets
            return targets

        def analyze_statement(details, file_index):
            dependencies = tuple(details.get("dependencies", []))
            assigned_vars = tuple(details.get("assigned_vars", []))
            line_no = self._coerce_line_number(details.get("line_no"))
            stmt_node = self._get_statement_node(file_index, line_no)
            cache_key = (stmt_node, dependencies, assigned_vars, line_no)
            cached_statement = statement_analysis_cache.get(cache_key)
            if cached_statement is not None:
                return cached_statement

            if stmt_node is None:
                statement = {
                    "stmt_kind": "Unknown",
                    "result_kind": "assign" if assigned_vars else "none",
                    "result_targets": list(assigned_vars),
                    "callsites": [],
                    "residual_dependencies": list(dependencies),
                }
            else:
                statement = RuntimeStatementAnalyzer(dependencies, assigned_vars).analyze(stmt_node)

            statement_analysis_cache[cache_key] = statement
            return statement

        def make_pending_line(item, frame_state):
            details = item.get("details", {})
            file_path = details.get("file_path")
            file_id = file_ids.get(file_path)
            if not file_id:
                return None

            scope = details.get("func") or "<module>"
            line_no = self._coerce_line_number(details.get("line_no"))
            file_index = source_index.get(file_path, {})
            statement = analyze_statement(details, file_index)
            target_ids = ensure_targets(
                details.get("assigned_vars", []),
                scope,
                file_id,
                line_no,
                file_index,
            )
            callsites = []
            for callsite in statement.get("callsites", []):
                callsites.append(
                    {
                        **callsite,
                        "matched": False,
                        "file_path": file_path,
                        "file_id": file_id,
                        "scope": scope,
                        "line_no": line_no,
                        "target_ids": target_ids,
                        "parent_frame_id": frame_state.get("frame_id", ""),
                    }
                )

            return {
                "file_path": file_path,
                "file_id": file_id,
                "scope": scope,
                "line_no": line_no,
                "stmt_kind": statement.get("stmt_kind", "Unknown"),
                "is_return": statement.get("result_kind") == "return",
                "residual_dependencies": statement.get("residual_dependencies", []),
                "target_ids": target_ids,
                "callsites": callsites,
                "matched_return_sources": set(),
            }

        def flush_pending_line(pending, frame_state):
            if pending is None:
                return

            line_no = pending["line_no"]
            file_path = pending["file_path"]
            file_id = pending["file_id"]
            scope = pending["scope"]
            stmt_kind = pending["stmt_kind"]
            target_ids = list(pending["target_ids"].values())
            residual_source_ids = resolve_many(
                pending["residual_dependencies"],
                scope,
                file_path,
                file_id,
                line_no,
            )

            if target_ids:
                for target_id in target_ids:
                    for source_id in residual_source_ids:
                        add_edge(source_id, target_id, line_no)
                if residual_source_ids or stmt_kind in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}:
                    add_step(
                        frame_state.get("frame_id", ""),
                        "def" if stmt_kind in {"FunctionDef", "AsyncFunctionDef", "ClassDef"} else "as",
                        line_no,
                        residual_source_ids,
                        target_ids,
                        {"stmt": stmt_kind},
                    )

            if not target_ids and residual_source_ids:
                op = "cond" if stmt_kind in {"If", "While", "For", "AsyncFor", "With", "AsyncWith"} else "use"
                add_step(
                    frame_state.get("frame_id", ""),
                    op,
                    line_no,
                    residual_source_ids,
                    [],
                    {"stmt": stmt_kind},
                )

            total_return_sources = set(pending["matched_return_sources"])
            for callsite in pending["callsites"]:
                if callsite["matched"]:
                    continue
                fallback_source_ids = resolve_many(
                    callsite.get("consumed_dependencies") or callsite.get("callable_dependencies"),
                    scope,
                    file_path,
                    file_id,
                    line_no,
                )
                if callsite["result_kind"] == "return":
                    total_return_sources.update(fallback_source_ids)
                elif target_ids and fallback_source_ids:
                    for target_id in target_ids:
                        for source_id in fallback_source_ids:
                            add_edge(source_id, target_id, line_no)
                    add_step(
                        frame_state.get("frame_id", ""),
                        "bind",
                        line_no,
                        fallback_source_ids,
                        target_ids,
                        {"stmt": stmt_kind, "callee": callsite.get("callee_label"), "fallback": True},
                    )

            if pending["is_return"]:
                total_return_sources.update(residual_source_ids)
                if total_return_sources:
                    frame_state["return_sources"].update(total_return_sources)
                add_step(
                    frame_state.get("frame_id", ""),
                    "ret",
                    line_no,
                    sorted(total_return_sources),
                    [],
                    {"stmt": stmt_kind},
                )

            for var_name, symbol_id in pending["target_ids"].items():
                latest_symbol[(file_id, scope, var_name)] = symbol_id

        def match_callsite(pending, func_name):
            if pending is None:
                return None
            for callsite in pending["callsites"]:
                if not callsite["matched"] and callsite.get("callee_hint") == func_name:
                    callsite["matched"] = True
                    return callsite
            for callsite in pending["callsites"]:
                if not callsite["matched"]:
                    callsite["matched"] = True
                    return callsite
            return None

        def process_call_event(call_event, parent_frame_state, callsite=None):
            nonlocal next_frame_id

            details = call_event.get("details", {})
            file_path = details.get("file_path")
            file_id = file_ids.get(file_path)
            if not file_id:
                return {"frame_id": "", "return_sources": set()}

            frame_id = f"c{next_frame_id}"
            next_frame_id += 1
            scope = details.get("func") or "<module>"
            depth = details.get("depth", 0)
            call_line = callsite["line_no"] if callsite else 0
            flow["frames"][frame_id] = [
                parent_frame_state.get("frame_id", ""),
                scope,
                file_id,
                depth,
                call_line,
            ]

            frame_state = {
                "frame_id": frame_id,
                "scope": scope,
                "file_path": file_path,
                "file_id": file_id,
                "return_sources": set(),
            }

            callable_source_ids = []
            if callsite is not None:
                callable_source_ids = resolve_many(
                    callsite.get("callable_dependencies"),
                    callsite["scope"],
                    callsite["file_path"],
                    callsite["file_id"],
                    callsite["line_no"],
                )
                add_step(
                    parent_frame_state.get("frame_id", ""),
                    "call",
                    callsite["line_no"],
                    callable_source_ids,
                    [],
                    {"callee": frame_id, "func": scope},
                )

                child_file_index = source_index.get(file_path, {})
                param_order = child_file_index.get("param_order_by_scope", {}).get(scope, [])
                param_lines = child_file_index.get("params_by_scope", {}).get(scope, {})
                arg_read_ids = []
                arg_write_ids = []
                for index, param_name in enumerate(param_order):
                    if index >= len(callsite["arg_dependency_groups"]):
                        break
                    source_ids = resolve_many(
                        callsite["arg_dependency_groups"][index],
                        callsite["scope"],
                        callsite["file_path"],
                        callsite["file_id"],
                        callsite["line_no"],
                    )
                    if not source_ids:
                        continue
                    param_symbol = ensure_symbol(
                        "param",
                        param_name,
                        scope,
                        file_id,
                        param_lines.get(param_name, 0),
                    )
                    latest_symbol[(file_id, scope, param_name)] = param_symbol
                    arg_write_ids.append(param_symbol)
                    arg_read_ids.extend(source_ids)
                    for source_id in source_ids:
                        add_edge(source_id, param_symbol, callsite["line_no"], kind="arg")

                if arg_read_ids or arg_write_ids:
                    add_step(
                        parent_frame_state.get("frame_id", ""),
                        "arg",
                        callsite["line_no"],
                        arg_read_ids,
                        arg_write_ids,
                        {"callee": frame_id},
                    )

            process_event_list(details.get("daughter_stack", []), frame_state)

            if callsite is not None:
                bound_source_ids = sorted(frame_state["return_sources"]) or callable_source_ids
                if callsite["result_kind"] == "return":
                    parent_frame_state["return_sources"].update(bound_source_ids)
                else:
                    target_ids = list(callsite["target_ids"].values())
                    if bound_source_ids and target_ids:
                        for target_id in target_ids:
                            for source_id in bound_source_ids:
                                add_edge(
                                    source_id,
                                    target_id,
                                    callsite["line_no"],
                                    kind="ret" if frame_state["return_sources"] else None,
                                )
                            for source_id in callable_source_ids:
                                add_edge(source_id, target_id, callsite["line_no"])
                        add_step(
                            parent_frame_state.get("frame_id", ""),
                            "bind",
                            callsite["line_no"],
                            bound_source_ids,
                            target_ids,
                            {
                                "callee": frame_id,
                                "stmt": callsite["result_kind"],
                                "kind": "ret" if frame_state["return_sources"] else "fallback",
                            },
                        )

            return {"frame_id": frame_id, "return_sources": frame_state["return_sources"]}

        def process_event_list(events, frame_state):
            pending = None
            for item in events:
                details = item.get("details", {})
                event_type = item.get("type")

                if event_type == "LINE":
                    flush_pending_line(pending, frame_state)
                    pending = make_pending_line(item, frame_state)
                    continue

                if event_type == "CALL":
                    callsite = match_callsite(pending, details.get("func"))
                    if callsite is None:
                        flush_pending_line(pending, frame_state)
                        pending = None
                    process_call_event(item, frame_state, callsite=callsite)
                    continue

                if event_type == "EXCEPTION":
                    flush_pending_line(pending, frame_state)
                    pending = None
                    add_step(
                        frame_state.get("frame_id", ""),
                        "exc",
                        self._coerce_line_number(details.get("line_no")),
                        [],
                        [],
                        {
                            "exception_type": details.get("exception_type"),
                            "stmt": "EXCEPTION",
                        },
                    )
                    continue

                flush_pending_line(pending, frame_state)
                pending = None

            flush_pending_line(pending, frame_state)

        root_state = {
            "frame_id": "",
            "scope": "<root>",
            "file_path": "",
            "file_id": "",
            "return_sources": set(),
        }
        process_event_list(self.call_stack.get("execution_stack", []), root_state)

        graph["edges"] = [
            [src, dst, kind, line_no, hits]
            for (src, dst, kind, line_no), hits in sorted(
                edge_hits.items(),
                key=lambda item: (item[0][3], item[0][0], item[0][1], item[0][2]),
            )
        ]
        graph["paths"] = self._build_paths(graph["edges"], graph["symbols"])

        self.dependency_by_file = graph
        self.flow_trace = flow
        self._artifact_cache = {"graph": graph, "flow": flow}
        return self._artifact_cache

    def parse_dependency(self):
        """
        Parse the execution stack into a compact dependency graph that is
        optimized for downstream LLM consumption.
        """
        return self._build_semantic_artifacts()["graph"]

    def parse_flow_trace(self):
        """Parse the execution stack into a compact runtime flow trace."""
        return self._build_semantic_artifacts()["flow"]
