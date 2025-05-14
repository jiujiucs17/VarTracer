import ast
import json

class LineDependencyAnalyzer(ast.NodeVisitor):
    def __init__(self, local_vars, global_vars):
        self.local_vars = local_vars
        self.global_vars = global_vars
        self.dependencies = set()  # 依赖的变量和函数
        self.assigned_vars = set()  # 被赋值的变量

    def visit_Name(self, node):
        """处理变量名"""
        if isinstance(node.ctx, ast.Load):  # 变量被加载（依赖）
            if node.id in self.local_vars or node.id in self.global_vars:
                self.dependencies.add(node.id)
        elif isinstance(node.ctx, ast.Store):  # 变量被存储（赋值）
            self.assigned_vars.add(node.id)

    def visit_Attribute(self, node):
        """处理属性访问（如 obj.method）"""
        # 递归访问对象部分
        self.visit(node.value)
        # 添加属性名
        if isinstance(node.value, ast.Name) and node.value.id in self.local_vars:
            self.dependencies.add(f"{node.value.id}.{node.attr}")

    def visit_Call(self, node):
        """处理函数调用（如 obj.method(x)）"""
        # 处理函数名或方法名
        self.visit(node.func)
        # 处理函数参数
        for arg in node.args:
            self.visit(arg)

    def visit_Assign(self, node):
        """处理赋值语句（如 z = obj.method(x)）"""
        # 处理左侧赋值目标
        for target in node.targets:
            self.visit(target)
        # 处理右侧表达式
        self.visit(node.value)

    def analyze(self, code_line):
        """分析单行代码"""
        try:
            tree = ast.parse(code_line, mode='exec')
            self.visit(tree)
        except SyntaxError:
            pass  # 忽略语法错误
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
                mapping variable names to their dependencies.
        """
        dependencies_by_file = {}
        call_stack_items = self.call_stack.get("execution_stack", [])

        def _traverse_stack(stack, file_path, dependencies_by_file):
            """
            Recursively traverse the call stack and collect dependencies.

            Args:
                stack (list): The stack to traverse.
                file_path (str): The file path to match.
                dependencies_by_file (dict): The dictionary to update with dependencies.
            """
            for item in stack:
                details = item.get("details", {})
                line_no = details.get("line_no", None)
                if details.get("file_path") == file_path:
                    # Extract assigned variables and dependencies
                    assigned_vars = details.get("assigned_vars", [])
                    dependencies = details.get("dependencies", [])
                    for var in assigned_vars:

                        for dep in dependencies:
                            # 如果变量已经存在于字典中，更新它的依赖
                            # print(f"var: {var}, dep: {dep}, line_no: {line_no}")
                            
                            if var in dependencies_by_file[file_path]:
                                # 如果变量已有这个依赖，但在新的行号处产生了新的依赖，将新的依赖添加到值的列表中
                                if dep in dependencies_by_file[file_path][var] and line_no not in dependencies_by_file[file_path][var][dep]:
                                        dependencies_by_file[file_path][var][dep].append(line_no)
                                else:
                                    dependencies_by_file[file_path][var][dep] = [line_no]     
                            # 否则，添加新的依赖
                            else:
                                dependencies_by_file[file_path][var] = {dep: [line_no]}

                # Recursively traverse nested daughter_stack if present
                if "daughter_stack" in details:
                    _traverse_stack(details["daughter_stack"], file_path, dependencies_by_file)

        # Iterate over each file in self.files
        for file_path in self.files:
            dependencies_by_file[file_path] = {}
            # Start recursive traversal of the call stack
            _traverse_stack(call_stack_items, file_path, dependencies_by_file)

        self.dependency_by_file = dependencies_by_file
        return dependencies_by_file

    