import sys
import os
import linecache
import sysconfig
import pkgutil
import json
from datetime import datetime

from .ASTParser import LineDependencyAnalyzer, DependencyTree
from .Utilities import safe_serialize, create_event, FrameSnapshot


class VarTracer:
    def __init__(self, only_project_root=None, clean_stdlib=True, ignore_module_func=False, verbose=False):
        
        # Initialize the VTracer instance
        self.raw_logs = []
        self.last_filename = None  
        self.log_trace_progess = True  

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
        # 提取出当前 module name 的顶层名字
        module_name = frame.f_globals.get('__name__', None)
        root_module = module_name.split('.')[0] if module_name else None
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
    
    def exec_stack_txt(self, output_path=None, output_name="VTrace_exec_stack.txt", shorten_path=True):

        result_lines = []
        indent_level = 0

        for record in self.raw_logs:
            frame = record['frame']  # FrameSnapshot 实例
            event = record['event']
            arg = record['arg']

            filename = os.path.abspath(frame.file_name)
            lineno = frame.line_no
            funcname = frame.function_name
            package = self._get_package_name(frame)
            module = self._get_module_name(frame)
            line_content = linecache.getline(filename, lineno).strip()
            indent = '    ' * indent_level

            display_filename = filename if not shorten_path else self._shorten_path(filename)
            file_info = f"MODULE '{module}' | FILE '{display_filename}' | FUNC '{funcname}'()"

            if event == 'call':
                result_lines.append(f"{indent}CALL → {file_info}")
                indent_level += 1
            elif event == 'return':
                indent_level = max(indent_level - 1, 0)
                result_lines.append(f"{indent}RETURN ← {file_info}")
            elif event == 'line':
                # 分析依赖关系
                analysis_result = self._analyze_dependencies(line_content, frame.locals, frame.globals)
                dependencies = analysis_result.get("dependencies", set())
                assigned_vars = analysis_result.get("assigned_vars", set())

                result_lines.append(f"{indent}LINE - {file_info} | LINE {lineno}")
                result_lines.append(f"{indent}       {line_content}")
                result_lines.append(f"{indent}       locals: { {k: safe_serialize(v) for k, v in frame.locals.items()} }")
                result_lines.append(f"{indent}       globals: { {k: safe_serialize(v) for k, v in frame.globals.items()} }")
                result_lines.append(f"{indent}       dependencies: {dependencies}")
                result_lines.append(f"{indent}       assigned_vars: {assigned_vars}")
            elif event == 'exception':
                exc_type, exc_value, _ = arg
                result_lines.append(f"{indent}EXCEPTION in {file_info}")
                result_lines.append(f"{indent}       {line_content}")
                result_lines.append(f"{indent}       type: {exc_type.__name__}, value: {safe_serialize(exc_value)}")

            else:
                # 其他事件类型，仅保存事件类型和提示：“this type of event is not supported”
                result_lines.append(f"{indent}UNSUPPORTED EVENT - {file_info}")
                result_lines.append(f"{indent}       event_type: {event}")
                result_lines.append(f"{indent}       VTrace message: This type of event is not supported.")

        # Add a timestamp to the output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_lines.insert(0, f"Trace started at {timestamp}")

        output = '\n'.join(result_lines)

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_name)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            if self.verbose:
                print(f"Txt execution stack saved to '{output_file}'")

        return output
    

    def exec_stack_json(self, output_path=None, output_name="VTrace_exec_stack.json"):
        """
        根据 raw_logs 生成 JSON 格式的执行堆栈，所有数据均保存为字符串。
        """

        def process_scope(logs):
            """递归处理作用域，生成嵌套的 JSON 堆栈"""
            stack = []
            while logs:
                record = logs.pop(0)
                frame = record['frame']
                event = record['event']
                arg = record['arg']

                filename = os.path.abspath(frame.file_name)
                lineno = frame.line_no
                funcname = frame.function_name
                module = self._get_module_name(frame)
                line_content = linecache.getline(filename, lineno).strip()

                # display_filename = filename if not shorten_path else self._shorten_path(filename)

                # 构造基础信息
                base_info = {
                    "module": safe_serialize(module),
                    "file_path": safe_serialize(filename),
                    "func": safe_serialize(funcname),
                }

                if event == 'call':
                    # CALL 事件，进入新作用域
                    call_event = create_event("CALL", base_info)
                    call_event["details"]["daughter_stack"] = process_scope(logs)
                    stack.append(call_event)
                elif event == 'return':
                    # RETURN 事件，退出当前作用域
                    return_event = create_event("RETURN", base_info)
                    stack.append(return_event)
                    break
                elif event == 'line':
                    # LINE 事件，分析依赖关系
                    analysis_result = self._analyze_dependencies(line_content, frame.locals, frame.globals)
                    dependencies = analysis_result.get("dependencies", set())
                    assigned_vars = analysis_result.get("assigned_vars", set())

                    line_event = create_event(
                        "LINE",
                        {
                            **base_info,
                            "line_no": safe_serialize(lineno),
                            "line_content": safe_serialize(line_content),
                            "locals": {k: safe_serialize(v) for k, v in frame.locals.items()},
                            "globals": {k: safe_serialize(v) for k, v in frame.globals.items()},
                            "dependencies": [safe_serialize(dep) for dep in dependencies],
                            "assigned_vars": [safe_serialize(var) for var in assigned_vars],
                        },
                    )
                    stack.append(line_event)
                elif event == 'exception':
                    # EXCEPTION 事件
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
                    # 其他事件类型，仅保存事件类型和提示：“this type of event is not supported”
                    unsupported_event = create_event(
                        "UNSUPPORTED",
                        {
                            **base_info,
                            "event_type": safe_serialize(event),
                            "VTrace message": "This type of event is not supported.",
                        },
                    )
                    

            return stack

        # 开始处理 raw_logs
        logs_copy = self.raw_logs.copy()
        result = process_scope(logs_copy)

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_data = {"trace_started_at": safe_serialize(timestamp), "execution_stack": result}

        # 输出到文件或控制台
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_name)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            # print(f"JSON 堆栈已保存到 {output_file}")
            # print console log with english
            if self.verbose:
                print(f"Nested JSON call stack saved to '{output_file}'")

        return output_data

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


        

