import sys
import os
import linecache
import sysconfig
import pkgutil
import json
from tqdm import tqdm
from datetime import datetime

from .ASTParser import LineDependencyAnalyzer, DependencyTree
from .Utilities import *


class VarTracer:
    def __init__(self, only_project_root=None, clean_stdlib=True, ignore_module_func=False, verbose=False):
        
        # Initialize the VTracer instance
        self.raw_logs = []
        self.last_filename = None  
        self.log_trace_progess = True  
        self.exec_stack = None # save processed execution stack in json format
        self.dep_tree = None  # save the dependency tree in json format

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
                    analysis_result = self._analyze_dependencies(line_content, frame.locals, frame.globals)
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

    def dep_tree_txt(self, output_path, output_name="VTrace_dep_tree.txt", short_path=True):
        """
        生成格式清晰、接近表格的 txt 版本依赖树。output_path 必须提供。
        以文件分组，每个文件一个依赖表格。
        每个表格的第一列为变量名，第二列为定义行号，第三列为依赖子表格（依赖变量、first occurrence、co-occurrences）。
        """
        if output_path is None:
            raise ValueError("output_path must be provided.")
    
        # 如果 dep_tree 为空，先生成
        if self.dep_tree is None:
            self.dep_tree_json(output_path=None)
    
        dep_tree = self.dep_tree
        timestamp = dep_tree.get("trace_started_at", "")
    
        def shorten_path(path):
            path = os.path.abspath(path)
            parts = path.split(os.sep)
            if len(parts) <= 4:
                return path
            return os.sep.join([parts[0], parts[1], '...', parts[-2], parts[-1]])
    
        result_lines = []
        result_lines.append(f"Dependency tree generated at {timestamp}")
        result_lines.append("=" * 120)
    
        for file_path, var_dict in dep_tree.items():
            if file_path == "trace_started_at":
                continue
            file_disp = shorten_path(file_path) if short_path else file_path
            result_lines.append(f"\nFile: {file_disp}")
            result_lines.append("-" * 120)
            result_lines.append(f"{'Variable':<25} | {'Line':<8} | {'Dependencies':<80}")
            result_lines.append("-" * 120)
            if not var_dict:
                result_lines.append(f"{'-':<25} | {'-':<8} | {'-':<80}")
                continue
            for var_key, var_info in var_dict.items():
                var_name = var_info.get("variableName", var_key)
                line_no = str(var_info.get("lineNumber", "-"))
                results = var_info.get("results", {})
                if not results:
                    result_lines.append(f"{var_name:<25} | {line_no:<8} | {'-':<80}")
                    continue
                # 构建依赖子表格
                dep_lines = []
                dep_lines.append(f"{'Dep.Var':<20} | {'First Occurrence':<18} | {'Co-occurrences':<35}")
                dep_lines.append(f"{'-'*20} | {'-'*18} | {'-'*35}")
                for dep_var, dep_info in results.items():
                    first_occ = str(dep_info.get("first_occurrence", "-"))
                    co_occs = dep_info.get("co_occurrences", [])
                    if isinstance(co_occs, list):
                        co_occs_str = ", ".join(str(x) for x in co_occs) if co_occs else "-"
                    else:
                        co_occs_str = "-"
                    dep_lines.append(f"{dep_var:<20} | {first_occ:<18} | {co_occs_str:<35}")
                # 将依赖子表格合并为单个字符串，缩进显示
                dep_table_str = "\n    ".join(dep_lines)
                result_lines.append(f"{var_name:<25} | {line_no:<8} | \n    {dep_table_str}")
            result_lines.append("-" * 120)
    
        output = '\n'.join(result_lines)
    
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, output_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        if self.verbose:
            print(f"Txt dependency tree saved to '{output_file}'")
    
        return output

    def dep_tree_xlsx(self, output_path, output_name="VTrace_dep_tree.xlsx", short_path=True):
        """
        生成依赖树的 XLSX 文件。以文件分组，每个文件一个 sheet，每个 sheet 是依赖表格。
        第一列为变量名，第二列为定义行号，第三列为依赖变量，第四列为 first occurrence，第五列为 co-occurrences。
        Sheet 名如有重复自动编号，例如 __init__.py(1)、__init__.py(2)，并按字母表顺序排列。
        每个sheet第一行显示完整路径。主变量和行号纵向合并居中。所有表格加边框。
        """
        import xlsxwriter

        if output_path is None:
            raise ValueError("output_path must be provided.")

        # 如果 dep_tree 为空，先生成
        if self.dep_tree is None:
            self.dep_tree_json(output_path=None)

        dep_tree = self.dep_tree

        def shorten_path(path):
            path = os.path.abspath(path)
            parts = path.split(os.sep)
            if len(parts) <= 4:
                return path
            return os.sep.join([parts[0], parts[1], '...', parts[-2], parts[-1]])

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, output_name)
        workbook = xlsxwriter.Workbook(output_file)

        # 定义带边框的格式
        border_fmt = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        header_fmt = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        path_fmt = workbook.add_format({'italic': True, 'border': 1, 'align': 'left', 'valign': 'vcenter'})

        # 收集所有 sheet 信息并排序
        sheet_infos = []
        for file_path, var_dict in dep_tree.items():
            if file_path == "trace_started_at":
                continue
            file_disp = shorten_path(file_path) if short_path else file_path
            base_name = os.path.basename(file_disp)
            sheet_infos.append((base_name, file_path, var_dict))
        # 按 base_name 字母表顺序排序
        sheet_infos.sort(key=lambda x: x[0].lower())

        # 检查 sheet 名称重复并编号
        sheet_name_count = {}
        for base_name, file_path, var_dict in sheet_infos:
            # Excel sheet name max length is 31
            sheet_name = base_name
            if len(sheet_name) > 31:
                sheet_name = sheet_name[-31:]
            orig_sheet_name = sheet_name
            count = sheet_name_count.get(orig_sheet_name, 0)
            while sheet_name in sheet_name_count:
                count += 1
                suffix = f"({count})"
                # 保证加编号后不超长
                if len(orig_sheet_name) + len(suffix) > 31:
                    sheet_name = orig_sheet_name[:31 - len(suffix)] + suffix
                else:
                    sheet_name = orig_sheet_name + suffix
            sheet_name_count[orig_sheet_name] = count
            sheet_name_count[sheet_name] = 0  # 标记已用
            worksheet = workbook.add_worksheet(sheet_name)

            # 第一行写完整路径
            worksheet.write(0, 0, file_path, path_fmt)

            # 第二行写表头
            worksheet.write(1, 0, "Variable", header_fmt)
            worksheet.write(1, 1, "Line", header_fmt)
            worksheet.write(1, 2, "Dep.Var", header_fmt)
            worksheet.write(1, 3, "First Occurrence", header_fmt)
            worksheet.write(1, 4, "Co-occurrences", header_fmt)

            row = 2
            if not var_dict:
                for col in range(5):
                    worksheet.write(row, col, "-", border_fmt)
                continue
            for var_key, var_info in var_dict.items():
                var_name = var_info.get("variableName", var_key)
                line_no = str(var_info.get("lineNumber", "-"))
                results = var_info.get("results", {})
                dep_items = list(results.items()) if results else []
                dep_count = len(dep_items) if dep_items else 1

                # 合并主变量和行号的单元格
                worksheet.merge_range(row, 0, row + dep_count - 1, 0, var_name, border_fmt)
                worksheet.merge_range(row, 1, row + dep_count - 1, 1, line_no, border_fmt)

                if not dep_items:
                    worksheet.write(row, 2, "-", border_fmt)
                    worksheet.write(row, 3, "-", border_fmt)
                    worksheet.write(row, 4, "-", border_fmt)
                    row += 1
                    continue

                for i, (dep_var, dep_info) in enumerate(dep_items):
                    first_occ = str(dep_info.get("first_occurrence", "-"))
                    co_occs = dep_info.get("co_occurrences", [])
                    if isinstance(co_occs, list):
                        co_occs_str = ", ".join(str(x) for x in co_occs) if co_occs else "-"
                    else:
                        co_occs_str = "-"
                    worksheet.write(row, 2, dep_var, border_fmt)
                    worksheet.write(row, 3, first_occ, border_fmt)
                    worksheet.write(row, 4, co_occs_str, border_fmt)
                    row += 1

        workbook.close()
        if self.verbose:
            print(f"XLSX dependency tree saved to '{output_file}'")
        return output_file

    def dep_tree_json(self, output_path=None, output_name="VTrace_dep_tree.json", show_progress=False):
            """
            生成依赖树的 JSON 文件。如果 self.dep_tree 为空，则根据 self.exec_stack 生成。
            返回 self.dep_tree。
            """
            if self.dep_tree is None:
                # 如果 exec_stack 为空，先生成
                if self.exec_stack is None:
                    self.exec_stack_json(output_path=None, show_progress=show_progress)
                # 构建依赖树
                dep_tree_parser = DependencyTree(call_stack=json.dumps(self.exec_stack))
                self.dep_tree = dep_tree_parser.parse_dependency()

            # 输出到文件
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                output_file = os.path.join(output_path, output_name)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.dep_tree, f, indent=4)
                if self.verbose:
                    print(f"Dependency tree JSON saved to '{output_file}'")

            return self.dep_tree
    
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


        

