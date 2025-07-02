import ast
import json

class LineDependencyAnalyzer(ast.NodeVisitor):
    def __init__(self, local_vars, global_vars):
        self.local_vars = set(local_vars)
        self.global_vars = set(global_vars)
        self.dependencies = set()  # 依赖的变量和函数
        self.assigned_vars = set()  # 被赋值的变量

    def visit_Name(self, node):
        # 变量名依赖和赋值
        if isinstance(node.ctx, ast.Load):
            if node.id in self.local_vars or node.id in self.global_vars:
                self.dependencies.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.assigned_vars.add(node.id)

    def visit_Attribute(self, node):
        # 递归处理链式属性 obj.a.b.c
        attr_chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            attr_chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            attr_chain.append(cur.id)
            full_attr = '.'.join(reversed(attr_chain))
            if isinstance(node.ctx, ast.Load):
                if cur.id in self.local_vars or cur.id in self.global_vars:
                    self.dependencies.add(full_attr)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                self.assigned_vars.add(full_attr)
        else:
            self.visit(cur)

    def visit_Call(self, node):
        # 支持链式调用 obj.method1().method2()
        # 新增：无论函数名是否在 local/global，最外层函数名都加入 dependencies
        def add_func_name_to_deps(func_node):
            # 递归找到最左侧的 Name
            if isinstance(func_node, ast.Name):
                self.dependencies.add(func_node.id)
            elif isinstance(func_node, ast.Attribute):
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
        for arg in node.args.args:
            self.local_vars.add(arg.arg)
        self.visit(node.body)

    def visit_ListComp(self, node):
        # 处理列表推导式
        for gen in node.generators:
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)
            if isinstance(gen.target, ast.Name):
                self.local_vars.add(gen.target.id)
            else:
                self.visit(gen.target)
        self.visit(node.elt)

    def visit_DictComp(self, node):
        for gen in node.generators:
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)
            if isinstance(gen.target, ast.Name):
                self.local_vars.add(gen.target.id)
            else:
                self.visit(gen.target)
        self.visit(node.key)
        self.visit(node.value)

    def visit_SetComp(self, node):
        for gen in node.generators:
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)
            if isinstance(gen.target, ast.Name):
                self.local_vars.add(gen.target.id)
            else:
                self.visit(gen.target)
        self.visit(node.elt)

    def visit_GeneratorExp(self, node):
        for gen in node.generators:
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)
            if isinstance(gen.target, ast.Name):
                self.local_vars.add(gen.target.id)
            else:
                self.visit(gen.target)
        self.visit(node.elt)

    def visit_FunctionDef(self, node):
        # 处理嵌套函数定义
        self.local_vars.add(node.name)
        for arg in node.args.args:
            self.local_vars.add(arg.arg)
        for stmt in node.body:
            self.visit(stmt)

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
    
    def parse_dependency(self):
        """
        Parse the dependencies for each file in self.files from self.call_stack.

        Returns:
            dict: A dictionary where keys are file paths and values are dictionaries
                mapping variable names (with scope chain and filename) to their dependencies.
        """
        import os

        dependencies_by_file = {}
        call_stack_items = self.call_stack.get("execution_stack", [])

        def _traverse_stack(stack, file_path, var_info, scope_chain):
            for item in stack:
                details = item.get("details", {})
                line_no = details.get("line_no", None)
                # if line_no == 0: line_no = 1  # 确保行号从1开始，以避免在后续使用 vscode extension 处理行号时遇到 0 行号为非法值的问题
                func_name = details.get("func", None)
                file_name = os.path.splitext(os.path.basename(details.get("file_path", "")))[0] if details.get("file_path") else ""
                # 构建新的作用域链
                # new_scope_chain = scope_chain.copy()
                new_scope_chain = []
                if func_name:
                    new_scope_chain.append(func_name)
                if details.get("file_path") == file_path:
                    assigned_vars = details.get("assigned_vars", [])
                    dependencies = details.get("dependencies", [])
                    errors = details.get("errors", {}) if "errors" in details else {}

                    for var in assigned_vars:
                        # 作用域链格式：filename.toplevelscopename.**.parentscopename.varname
                        scoped_var = ".".join([file_name] + new_scope_chain + [var]) if file_name else ".".join(new_scope_chain + [var])
                        if scoped_var not in var_info:
                            var_info[scoped_var] = {
                                "variableName": var,
                                "lineNumber": line_no,
                                "results": {} if not errors.get(var) else errors[var]
                            }
                        # 错误处理
                        if errors.get(var):
                            var_info[scoped_var]["results"] = errors[var]
                            continue
                        # 依赖处理
                        for dep in dependencies:
                            if isinstance(var_info[scoped_var]["results"], str):
                                continue
                            if dep not in var_info[scoped_var]["results"]:
                                var_info[scoped_var]["results"][dep] = {
                                    "first_occurrence": line_no,
                                    "co_occurrences": [line_no]
                                }
                            else:
                                if line_no not in var_info[scoped_var]["results"][dep]["co_occurrences"]:
                                    var_info[scoped_var]["results"][dep]["co_occurrences"].append(line_no)
                # 递归遍历 daughter_stack
                if "daughter_stack" in details:
                    _traverse_stack(details["daughter_stack"], file_path, var_info, new_scope_chain)

        for file_path in self.files:
            var_info = {}
            _traverse_stack(call_stack_items, file_path, var_info, [])
            # 对每个依赖的 co_occurrences 排序，去重
            for var in var_info:
                results = var_info[var]["results"]
                if isinstance(results, dict):
                    for dep in results:
                        results[dep]["co_occurrences"] = sorted(list(set(results[dep]["co_occurrences"])))
            # 第二步：修正 first_occurrence
            for var in var_info:
                results = var_info[var]["results"]
                if isinstance(results, dict):
                    for dep in results:
                        # dep 变量的 lineNumber
                        # dep 可能是未加作用域链的变量名，需要补全作用域链
                        dep_scoped = None
                        for candidate in var_info:
                            if candidate.endswith(f".{dep}") or candidate == dep:
                                dep_scoped = candidate
                                break
                        if dep_scoped and "lineNumber" in var_info[dep_scoped]:
                            results[dep]["first_occurrence"] = var_info[dep_scoped]["lineNumber"]
#
            dependencies_by_file[file_path] = var_info

        self.dependency_by_file = dependencies_by_file
        return dependencies_by_file