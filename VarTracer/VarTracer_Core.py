import sys
import os
import linecache
import sysconfig
import pkgutil
import json
import ast
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    class _NoOpTqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def tqdm(*args, **kwargs):
        return _NoOpTqdm()

from .ASTParser import DEPENDENCY_GRAPH_COMMENT, FLOW_TRACE_COMMENT, DependencyTree, LineDependencyAnalyzer
from .Utilities import *


class VarTracer:
    def __init__(self, only_project_root=None, clean_stdlib=True, ignore_module_func=False, verbose=False):
        
        # Initialize the VTracer instance
        self.raw_logs = []
        self.last_filename = None  
        self.log_trace_progess = True  
        self.exec_stack = None # save processed execution stack in json format
        self.dep_tree = None  # save the dependency tree in json format
        self.flow_trace = None  # save the runtime flow trace in json format
        self._statement_cache = {}
        self._internal_package_root = os.path.abspath(os.path.dirname(__file__))

        # Set the parameters
        self.only_project_root = os.path.abspath(only_project_root) if only_project_root else None
        self.clean_stdlib = clean_stdlib 
        self.ignore_module_funcs = ignore_module_func 
        self.verbose = verbose

        # Initialize the ignored modules 
        self.stdlibs = set(self._expand_module_names(self._get_standard_library_modules()))
        self.stdlibs.remove("__main__")
        self.frozen_modules = set(self._expand_module_names(sys.builtin_module_names)) 
        self.ignored_modules = self.stdlibs.union(self.frozen_modules) if self.clean_stdlib else self.frozen_modules 
        self.ignored_modules.add("_distutils_hack")

        # print(self.stdlibs)

        # Initialize the ignored functions
        self.ignored_funcs = set(['<module>']) if self.ignore_module_funcs else set()  


    def _get_package_name(self, frame_snapshot):
        """Get the package name from the frame snapshot."""
        if frame_snapshot.package_name:
            return frame_snapshot.package_name
        if frame_snapshot.module_name and frame_snapshot.module_name != '__main__':
            return frame_snapshot.module_name
        return '(unknown-package)'
    
    def _get_module_name(self, frame_snapshot):
        """Get the full module name for the given frame snapshot."""       
        if frame_snapshot.module_name and frame_snapshot.module_name != '__main__':
            # Use the module name directly if it's not '__main__'
            return frame_snapshot.module_name
        
        # If all else fails, return a placeholder for unknown modules
        return '(unknown-module)'
    
    def _expand_module_names(self, module_names):
        """Expand module names to include all submodules."""
        expanded = set()
        for name in module_names:
            parts = name.split('.')
            for i in range(1, len(parts) + 1):
                expanded.add(parts[i - 1])  # 加入每一层的单独模块名
            expanded.add(name.split('.')[-1])  # 加入最后一级模块名
        return expanded

    def _get_standard_library_modules(self):
        """Get all standard library modules."""
        stdlib_path = sysconfig.get_paths()["stdlib"]
        stdlib_modules = set()

        def collect_modules(path, prefix=''):
            for module_info in pkgutil.iter_modules([path]):
                name = prefix + module_info.name
                stdlib_modules.add(name)

                if module_info.ispkg:
                    # 构造子包路径
                    sub_path = os.path.join(path, module_info.name)
                    collect_modules(sub_path, prefix=name + '.')

        collect_modules(stdlib_path)
        return self._expand_module_names(stdlib_modules)
    
    def _shorten_path(self, path, max_len=60):
        path = os.path.abspath(path)
        parts = path.split(os.sep)

        if len(path) <= max_len:
            return path

        if len(parts) <= 3:
            return path  # 不需要缩短

        return os.sep.join([
            parts[0],              # 顶层目录（如 'C:' 或 '/'）
            parts[1],              # 第二层目录（如 'Users' 或 'home'）
            '...',                 # 中间省略
            parts[-3],             # 倒数第三部分（如 'Documents' 或 'my_project'）
            parts[-2],             # 倒数第二部分（目录）
            parts[-1]              # 文件名
        ])

    def _trace(self, frame, event, arg):
        filename = os.path.abspath(frame.f_code.co_filename)
        # 提取出当前 module name 的顶层名字
        module_name = frame.f_globals.get('__name__', None)
        root_module = module_name.split('.')[0] if module_name else None
        if module_name and module_name.startswith("VarTracer") and filename.startswith(self._internal_package_root):
            return None
        # 过滤掉标准库模块、frozen 模块和其他需要忽略的模块
        if root_module in self.ignored_modules:
            # print(f"Trace: Ignoring standard library module: {module_name}")
            return None
        
        # 如果 func name 在 ignored_funcs 中，则忽略该事件
        func_name = frame.f_code.co_name
        if func_name in self.ignored_funcs:
            # print("Ignoring function:", func_name)
            return None

        # 保存原始事件
        self.raw_logs.append({
            'frame': FrameSnapshot(frame),
            'event': event,
            'arg': arg
        })

        # # Curently disabled to speed up the trace, uncomment to enable
        # # 如果启用了 log_trace_progess，则在控制台动态显示文件名和行号
        # if self.log_trace_progess:
        #     file_name = os.path.basename(frame.f_code.co_filename)  # 获取文件名
        #     line_no = frame.f_lineno  # 获取行号
        #     message = f"Tracing: {file_name} - Line {line_no}"

        #     # 获取终端宽度
        #     try:
        #         terminal_width = os.get_terminal_size().columns
        #     except OSError:
        #         terminal_width = 80  # 默认宽度

        #     # 计算需要填充的空格数量
        #     padding = max(terminal_width - len(message), 0)
        #     sys.stdout.write(f"\r{message}{' ' * padding}")  # 动态更新
        #     sys.stdout.flush()

        return self._trace
    
    def _clean(self):
        """清理 raw_logs 中属于标准库模块的记录。"""
        stdlib_modules = self._get_standard_library_modules()
            
        # Expand and merge the ignore_modules and stdlib_modules sets
        ignore_modules = self.ignored_modules
        ignore_modules.update(stdlib_modules)  # Use update instead of add

        def is_stdlib(frame_snapshot):
            module_name = self._get_module_name(frame_snapshot)
            root_module = module_name.split('.')[0]
            return root_module in ignore_modules

        original_count = len(self.raw_logs)

        self.raw_logs = [
            record for record in self.raw_logs
            if not is_stdlib(record['frame'])
        ]
        cleaned_count = len(self.raw_logs)

        # print(f"清理完成：从 {original_count} 条日志中移除 {original_count - cleaned_count} 条标准库记录。")
        # print console log with english
        if self.verbose:
            print(f"Note that the execution log is cleaned: Removed {original_count - cleaned_count} standard Python library records from {original_count} logs. Initalize VTracer instance using 'VTracer(clean_stdlib=False)' to disable this feature.")

    def _analyze_dependencies(self, line_content, local_vars, global_vars):
        """
        使用 DependencyAnalyzer 分析代码行中的依赖关系。
        """
        analyzer = LineDependencyAnalyzer(local_vars, global_vars)
        return analyzer.analyze(line_content)

    def _get_statement_cache(self, filename):
        filename = os.path.abspath(filename)
        if filename not in self._statement_cache:
            cache_entry = {"stmt_nodes": []}
            try:
                with open(filename, 'r', encoding='utf-8') as handle:
                    source = handle.read()
                tree = ast.parse(source, filename=filename)
                cache_entry["stmt_nodes"] = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.stmt) and hasattr(node, "lineno") and hasattr(node, "end_lineno")
                ]
            except (OSError, SyntaxError):
                pass
            self._statement_cache[filename] = cache_entry
        return self._statement_cache[filename]

    def _get_statement_node(self, filename, lineno):
        stmt_nodes = self._get_statement_cache(filename).get("stmt_nodes", [])

        def statement_key(node):
            end_lineno = getattr(node, "end_lineno", node.lineno)
            end_col = getattr(node, "end_col_offset", 0)
            start_col = getattr(node, "col_offset", 0)
            return (end_lineno - node.lineno, end_col - start_col)

        starting_here = [node for node in stmt_nodes if node.lineno == lineno]
        if starting_here:
            return min(starting_here, key=statement_key)

        covering_line = [
            node for node in stmt_nodes
            if node.lineno <= lineno <= getattr(node, "end_lineno", node.lineno)
        ]
        if covering_line:
            return min(covering_line, key=statement_key)
        return None

    def _analyze_line_event(self, filename, lineno, line_content, local_vars, global_vars):
        stmt_node = self._get_statement_node(filename, lineno)
        if stmt_node is not None:
            analyzer = LineDependencyAnalyzer(local_vars, global_vars)
            return analyzer.analyze_node(stmt_node)
        return self._analyze_dependencies(line_content, local_vars, global_vars)


    def start(self):
        # 可以使用pipe来实现多进程间传递数据，并且多平台通用，来提升trace速度。
        self.raw_logs.clear()
        self.last_filename = None

        # Install the global tracing function
        sys.settrace(self._trace)

        # Explicitly set the trace function for the current frame
        parent_frame = sys._getframe().f_back
        while parent_frame:
            parent_frame.f_trace = self._trace
            parent_frame = parent_frame.f_back
            # print(f"Tracing function set for parent frame, frame: {parent_frame}")

    def stop(self):
        sys.settrace(None)
        # # this self.clean_stdlib is not needed here, because we have already cleaned the raw_logs in the trace method
        # if self.clean_stdlib:
        #     self._clean()  # 清理标准库模块的记录

    def raw_result(self, output_dir=None):
        result_lines = []

        for record in self.raw_logs:
            frame = record['frame']  # StaticFrame 实例
            event = record['event']

            result_lines.append(
                f"{event.upper()} - {frame.file_name}:{frame.line_no} - {frame.function_name}"
            )

        output = '\n'.join(result_lines)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"VTrace_raw_output.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(output)
            # print(f"追踪结果已保存到 {path}")
            # print console log with english
            if self.verbose:
                print(f"Raw trace result saved to '{path}'")
        else:
            if self.verbose:
                print(output)

        return output
    
    def exec_stack_txt(self, output_path, output_name="VTrace_exec_stack.txt", short_path=True):
        """
        生成格式清晰、缩进整齐的 txt 版本执行堆栈。output_path 必须提供。
        优先使用 self.exec_stack，没有则自动生成。
        file, func, line 字段均小写且各自独占一行。
        参数 short_path 控制文件路径显示方式。
        """
        if output_path is None:
            raise ValueError("output_path must be provided.")
    
        # 如果 exec_stack 为空，先生成
        if self.exec_stack is None:
            self.exec_stack_json(output_path=None, show_progress=False)
    
        exec_stack = self.exec_stack.get("execution_stack", [])
        timestamp = self.exec_stack.get("trace_started_at", "")
    
        result_lines = []
        result_lines.append(f"Trace started at {timestamp}")
    
        def shorten_path(path):
            path = os.path.abspath(path)
            parts = path.split(os.sep)
            if len(parts) <= 4:
                return path
            # 例如: /a/b/.../y/z/filename.py
            return os.sep.join([parts[0], parts[1], '...', parts[-2], parts[-1]])
    
        def format_event(event, indent_level=0):
            indent = "    " * indent_level
            lines = []
            event_type = event.get("type", "")
            event_type = event_type.upper()
            details = event.get("details", {})
    
            # 文件路径处理
            file_path = details.get('file_path', '')
            if short_path and file_path:
                file_path_disp = shorten_path(file_path)
            else:
                file_path_disp = file_path


            lines.append(f"{indent}{indent_level}--------------------------------------------------")
            if event_type == "CALL":
                lines.append(f"{indent}|CALL - '{details.get('module', '(unknown-module)')}'")
                lines.append(f"{indent}|       file: {file_path_disp}")
                lines.append(f"{indent}|       func: {details.get('func')}")
                daughter_stack = details.get("daughter_stack", [])
                for sub_event in daughter_stack:
                    lines.extend(format_event(sub_event, indent_level + 1))
            elif event_type == "RETURN":
                lines.append(f"{indent}|RETURN - '{details.get('module', '(unknown-module)')}'")
                lines.append(f"{indent}|       file: {file_path_disp}")
                lines.append(f"{indent}|       func: {details.get('func')}")
            elif event_type == "LINE":
                lines.append(f"{indent}|LINE - '{details.get('module', '(unknown-module)')}'")
                lines.append(f"{indent}|       line: {details.get('line_no')}")
                lines.append(f"{indent}|       file: {file_path_disp}")
                lines.append(f"{indent}|       func: {details.get('func')}")
                lines.append(f"{indent}|       content: {details.get('line_content')}")
            elif event_type == "EXCEPTION":
                lines.append(f"{indent}|EXCEPTION - '{details.get('module', '(unknown-module)')}'")
                lines.append(f"{indent}|       file: {file_path_disp}")
                lines.append(f"{indent}|       func: {details.get('func')}")
                lines.append(f"{indent}|       line: {details.get('line_no')}")
                lines.append(f"{indent}|       {details.get('line_content')}")
                lines.append(f"{indent}|       type: {details.get('exception_type')}, value: {details.get('exception_value')}")
            else:
                lines.append(f"{indent}|UNKNOWN EVENT: {event_type}")
                lines.append(f"{indent}|       file: {file_path_disp}")
                lines.append(f"{indent}|       func: {details.get('func')}")
                lines.append(f"{indent}|       event_type: {event_type}")
                lines.append(f"{indent}|       VTrace message: {details.get('VTrace message', 'This type of event is not supported.')}")
            return lines
    
        for event in exec_stack:
            result_lines.extend(format_event(event, indent_level=0))
    
        output = '\n'.join(result_lines)
    
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, output_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        if self.verbose:
            print(f"Txt execution stack saved to '{output_file}'")
    
        return output

    def exec_stack_json(self, output_path=None, output_name="VTrace_exec_stack.json", show_progress=False):
        """
        根据 raw_logs 生成 JSON 格式的执行堆栈，所有数据均保存为字符串。
        每个事件的 details 中都包含 depth 字段，表示调用深度（最浅为 0）。
        """

        def process_scope(logs, depth=0, pbar=None):
            """递归处理作用域，生成嵌套的 JSON 堆栈，并记录调用深度"""
            stack = []
            while logs:
                record = logs.pop(0)
                frame = record['frame']
                event = record['event']
                arg = record['arg']

                if pbar:
                    pbar.update(1)

                filename = os.path.abspath(frame.file_name)
                lineno = frame.line_no
                funcname = frame.function_name
                module = self._get_module_name(frame)
                line_content = linecache.getline(filename, lineno).strip()

                # 构造基础信息
                base_info = {
                    "module": safe_serialize(module),
                    "file_path": safe_serialize(filename),
                    "func": safe_serialize(funcname),
                    "depth": depth  # 新增 depth 字段
                }

                if event == 'call':
                    call_event = create_event("CALL", base_info)
                    call_event["details"]["daughter_stack"] = process_scope(logs, depth=depth+1, pbar=pbar)
                    stack.append(call_event)
                elif event == 'return':
                    return_event = create_event("RETURN", base_info)
                    stack.append(return_event)
                    break
                elif event == 'line':
                    analysis_result = self._analyze_line_event(filename, lineno, line_content, frame.locals, frame.globals)
                    dependencies = analysis_result.get("dependencies", set())
                    assigned_vars = analysis_result.get("assigned_vars", set())

                    line_event = create_event(
                        "LINE",
                        {
                            **base_info,
                            "line_no": safe_serialize(lineno),
                            "line_content": safe_serialize(line_content),
                            # "locals": {k: safe_serialize(v) for k, v in frame.locals.items()},
                            # "globals": {k: safe_serialize(v) for k, v in frame.globals.items()},
                            "dependencies": [safe_serialize(dep) for dep in dependencies],
                            "assigned_vars": [safe_serialize(var) for var in assigned_vars],
                        },
                    )
                    stack.append(line_event)
                elif event == 'exception':
                    exc_type, exc_value, _ = arg
                    exception_event = create_event(
                        "EXCEPTION",
                        {
                            **base_info,
                            "line_no": safe_serialize(lineno),
                            "line_content": safe_serialize(line_content),
                            "exception_type": safe_serialize(exc_type.__name__),
                            "exception_value": safe_serialize(exc_value),
                        },
                    )
                    stack.append(exception_event)
                else:
                    unsupported_event = create_event(
                        "UNSUPPORTED",
                        {
                            **base_info,
                            "event_type": safe_serialize(event),
                            "VTrace message": "This type of event is not supported.",
                        },
                    )
                    stack.append(unsupported_event)
            return stack

        # 开始处理 raw_logs
        logs_copy = self.raw_logs.copy()
        pbar = tqdm(total=len(logs_copy), desc="Processing logs", unit="log") if show_progress else None
        result = process_scope(logs_copy, depth=0, pbar=pbar if show_progress else None)
        if pbar:
            pbar.close()

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_data = {"trace_started_at": safe_serialize(timestamp), "execution_stack": result}

        # 输出到文件或控制台
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_name)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            if self.verbose:
                print(f"Nested JSON call stack saved to '{output_file}'")

        self.exec_stack = output_data  # 保存到实例变量中
        return output_data

    def dep_tree_json(self, output_path=None, output_name="VTrace_dep_tree.json", show_progress=False):
        """
        生成面向 LLM 的依赖图 JSON 文件。如果 self.dep_tree 为空，则根据
        self.exec_stack 生成。
        """
        if self.dep_tree is None:
            if self.exec_stack is None:
                self.exec_stack_json(output_path=None, show_progress=show_progress)
            dep_tree_parser = DependencyTree(call_stack=json.dumps(self.exec_stack))
            self.dep_tree = dep_tree_parser.parse_dependency()
            self.flow_trace = dep_tree_parser.parse_flow_trace()
        elif "_comment" not in self.dep_tree:
            self.dep_tree = {
                "_comment": DEPENDENCY_GRAPH_COMMENT,
                **self.dep_tree,
            }

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_name)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dep_tree, f, indent=4)
            if self.verbose:
                print(f"Dependency tree JSON saved to '{output_file}'")

        return self.dep_tree

    def flow_trace_json(self, output_path=None, output_name="VTrace_flow_trace.json", show_progress=False):
        """
        生成面向 LLM 的运行时 flow trace JSON 文件。如果 self.flow_trace 为空，则根据
        self.exec_stack 生成。
        """
        if self.flow_trace is None:
            if self.exec_stack is None:
                self.exec_stack_json(output_path=None, show_progress=show_progress)
            trace_parser = DependencyTree(call_stack=json.dumps(self.exec_stack))
            self.dep_tree = trace_parser.parse_dependency()
            self.flow_trace = trace_parser.parse_flow_trace()
        elif "_comment" not in self.flow_trace:
            self.flow_trace = {
                "_comment": FLOW_TRACE_COMMENT,
                **self.flow_trace,
            }

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_name)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.flow_trace, f, indent=4)
            if self.verbose:
                print(f"Flow trace JSON saved to '{output_file}'")

        return self.flow_trace

    def dep_tree_edgelist(self, output_path=None, output_name="VTrace_dep_tree.edgelist", show_progress=False):
        """
        将 dep_tree_json() 进一步压缩为便于直接发送给 LLM 的 edge-list 文本。
        """
        dep_tree = self.dep_tree_json(output_path=None, show_progress=show_progress)
        output_text = _dep_tree_to_edgelist_text(dep_tree)

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_name)
            with open(output_file, "w", encoding="utf-8") as handle:
                handle.write(output_text)
            if self.verbose:
                print(f"Dependency edge-list saved to '{output_file}'")

        return output_text

    def edgelist(self, output_path=None, output_name="VTrace_dep_tree.edgelist", show_progress=False):
        """Alias for dep_tree_edgelist()."""
        return self.dep_tree_edgelist(
            output_path=output_path,
            output_name=output_name,
            show_progress=show_progress,
        )
    
class FileVTracer:
    def __init__(self, filepath, verbose=False):
        """
        Initialize the FileVTracer instance with a file path.
        Checks if the file exists and is a Python file.
        """
        self.filepath = filepath
        self.verbose = verbose
        self.vt = VarTracer(verbose=self.verbose)
        self.file_content = None

        if self._validate_file():
            self.file_content = self._read_file()

    def _validate_file(self):
        """
        Validate the provided file path.
        Raise an error if the file does not exist or is not a Python file.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"The file '{self.filepath}' does not exist.")
        
        if not self.filepath.endswith('.py'):
            raise ValueError(f"The file '{self.filepath}' is not a Python file. Please provide a '.py' file.")

        # print(f"File '{self.filepath}' is valid and ready for tracing.")
        return True

    def _read_file(self):
        """
        Read the content of the file.
        """
        with open(self.filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    
    def trace(self, output_dir=None):
        """
        Use VTracer to trace the execution of the file content.
        """
        if not self.file_content:
            raise ValueError("No file content to trace. Ensure the file is valid and readable.")

        # Start the VTracer
        self.vt.start()

        try:
            # Execute the file content in a controlled environment
            exec(self.file_content, {})
        except Exception as e:
            if self.verbose:
                print(f"An error occurred during execution: {e}")
        finally:
            # Stop the VTracer
            self.vt.stop()

        # Output the raw trace results
        if output_dir:
            self.vt.raw_result(output_dir=output_dir)
            self.vt.exec_stack_txt(output_path=output_dir)
            self.vt.exec_stack_json(output_path=output_dir)
        else:
            self.vt.raw_result()
            self.vt.exec_stack_txt()
            self.vt.exec_stack_json()

        return self.vt.exec_stack_json(output_path=output_dir)



if __name__ == "__main__":
    from pathlib import Path
    # from test_code import playground as pg
    output_dir = os.path.join(Path(__file__).resolve().parent, "trace_output")
    
    vt = VarTracer(clean_stdlib=True)
    vt.start()

    from test_code import playground_2 as pg2
    pg2.main()

    vt.stop()
    vt.raw_result(output_dir=output_dir)
    vt.exec_stack_txt(output_path=output_dir)
    exec_stack_json = vt.exec_stack_json(output_path=output_dir)

    dep_tree = DependencyTree(call_stack=json.dumps(exec_stack_json))
    dep_dic = dep_tree.parse_dependency()

    # # 打印dep_dic这个字典中的所有内容
    # print("\n\nDependency Dictionary:")
    # for key, value in dep_dic.items():
    #     print(f"{key}: {value}")

    # print("\n\nstart to trace the file")
    # fvt = FileVTracer(filepath=os.path.join(Path(__file__).resolve().parent, "test_code", "playground.py"))
    # fvt.trace(output_dir=output_dir)
    # print("end to trace the file\n\n")


        
