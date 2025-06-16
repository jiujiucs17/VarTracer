import inspect
import os
import shutil
import subprocess
import json

def safe_serialize(obj):
        """将对象转换为字符串表示"""
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"
        
def create_event(event_type, base_info, extra_info=None):
    """创建一个事件字典"""
    event = {"type": event_type, "details": base_info}
    if extra_info:
        event["details"].update(extra_info)
    return event

class FrameSnapshot:
    def __init__(self, frame):
        self.file_name = frame.f_code.co_filename
        self.function_name = frame.f_code.co_name
        self.line_no = frame.f_lineno
        self.locals = frame.f_locals.copy()
        self.globals = {k: str(v) for k, v in frame.f_globals.items() if k in frame.f_code.co_names}
        self.code_context = self._get_code_context(frame)

        self.package_name = frame.f_globals.get('__package__', None)
        self.module_name = frame.f_globals.get('__name__', None) 

    def _get_code_context(self, frame, context_lines=2):
        """Get the source code context around the current line."""
        try:
            lines, start = inspect.getsourcelines(frame.f_code)
            index = frame.f_lineno - start
            lower = max(index - context_lines, 0)
            upper = min(index + context_lines + 1, len(lines))
            return [line.rstrip('\n') for line in lines[lower:upper]]
        except (OSError, TypeError):
            return []

    def to_dict(self):
        """Convert the FrameSnapshot to a dictionary for easier serialization."""
        return {
            "filename": self.file_name,
            "function_name": self.function_name,
            "line_no": self.line_no,
            "locals": self.locals,
            "globals": self.globals,
            "code_context": self.code_context,
        }

    def __repr__(self):

        """Return a string representation of the FrameSnapshot."""
        return f"<FrameSnapshot {self.function_name} at {self.file_name}:{self.line_no}>"
    
def extension_interface(file_path, print=False):
    """
    不直接修改 file_path 指向的文件，而是在同目录下创建一个新文件（文件名加(VarTracerTemp)），
    写入插桩后的内容，运行新文件，生成数据，最后删除新文件。
    输出结果中所有的"(VarTracerTemp)"字符串都会被去除。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.endswith('.py'):
        raise ValueError("The provided file is not a Python (.py) file.")

    file_dir = os.path.dirname(file_path)
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))
    temp_file_name = f"{file_base}(VarTracerTemp){file_ext}"
    temp_file_path = os.path.join(file_dir, temp_file_name)
    result_path = os.path.join(file_dir, "result.json")

    def remove_vartracertemp_str(obj):
        """递归地将对象中的'(VarTracerTemp)'字符串去除"""
        if isinstance(obj, str):
            return obj.replace("(VarTracerTemp)", "")
        elif isinstance(obj, dict):
            return {remove_vartracertemp_str(k): remove_vartracertemp_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [remove_vartracertemp_str(i) for i in obj]
        else:
            return obj

    try:
        # 读取原始文件内容
        with open(file_path, 'r') as f:
            original_content = f.readlines()

        # === 重要说明 ===
        # 假设：被读取的原始文件中的所有 import 语句都在文件开头，且没有穿插主代码。
        # 这样做的原因是：sys.settrace 必须在所有 import 语句之后调用，否则会递归追踪 importlib 导致爆栈。
        # 因此，我们扫描原始文件的前若干行，将所有以 'import ' 或 'from ' 开头的行视为 import 语句，
        # 并把与 VarTracer 相关的插桩代码插入到这些 import 语句之后、主代码之前。
        # 这样可以最大程度保证追踪主代码和第三方库调用，但不会递归追踪 import 过程。

        # 找到 import 语句的结束行号
        import_end = 0
        for idx, line in enumerate(original_content):
            striped = line.lstrip()
            if striped.startswith('import ') or striped.startswith('from '):
                import_end = idx + 1
            elif striped == '' or striped.startswith('#'):
                continue  # 跳过空行和注释
            else:
                break  # 第一个非 import/空行/注释的地方停止

        # 构造插桩内容
        vartracer_instrument = [
            "from VarTracer import *\n",
            "import json\n",
            "vt = VarTracer()\n",
            "vt.start()\n"
        ]
        # 插入到所有 import 语句之后
        modified_content = (
            original_content[:import_end] +
            vartracer_instrument +
            original_content[import_end:] +
            [
                "\n",
                "vt.stop()\n",
                "exec_stack_json = vt.exec_stack_json()\n",
                "dep_tree = DependencyTree(call_stack=json.dumps(exec_stack_json))\n",
                "dep_dic = dep_tree.parse_dependency()\n",
                "\n",
                "result_json = {\n",
                "    'exec_stack': exec_stack_json,\n",
                "    'dependency': dep_dic\n",
                "}\n",
                "\n",
                f"with open(r'{result_path}', 'w') as result_file:\n",
                "    json.dump(result_json, result_file)\n"
            ]
        )

        # 写入临时文件
        with open(temp_file_path, 'w') as f:
            f.writelines(modified_content)

        # 运行临时文件
        try:
            std_exe_result = subprocess.run(['python3', temp_file_path],
                                            check=True,
                                            capture_output=True,
                                            text=True)
        except subprocess.CalledProcessError as e:
            raise e

        with open(result_path, 'r') as result_file:
            result_json = json.load(result_file)

        # 修正 dependency 中 temp_file_path 指向脚本的变量行号
        def fix_dependency_line_numbers(dependency, target_path):
            if target_path in dependency:
                for var, info in dependency[target_path].items():
                    if "lineNumber" in info and info["lineNumber"].isdigit():
                        info["lineNumber"] = str(int(info["lineNumber"]) - 4)
                    if "results" in info and isinstance(info["results"], dict):
                        for res in info["results"].values():
                            if "first_occurrence" in res and res["first_occurrence"].isdigit():
                                res["first_occurrence"] = str(int(res["first_occurrence"]) - 4)
                            if "co_occurrences" in res and isinstance(res["co_occurrences"], list):
                                res["co_occurrences"] = [
                                    str(int(x) - 4) if x.isdigit() else x for x in res["co_occurrences"]
                                ]

        # 修正 execution_stack 中 temp_file_path 指向脚本的行号，以及删除最后一个行的 VarTracer 相关信息，即代码内容为“exec_stack_json = vt.exec_stack_json()”的元素
        def fix_stack_line_numbers(execution_stack, target_path):
            for frame in execution_stack:
                details = frame.get("details", {})
                if details.get("file_path") == target_path:
                    if "line_no" in details and str(details["line_no"]).isdigit():
                        details["line_no"] = str(int(details["line_no"]) - 4)
                    if "line_content" in details and str(details["line_content"]).startswith("exec_stack_json = vt.exec_stack_json()"):
                        execution_stack.remove(frame)

            

        target_path = temp_file_path
        if "dependency" in result_json:
            fix_dependency_line_numbers(result_json["dependency"], target_path)

        if "execution_stack" in result_json.get("exec_stack", {}):
            fix_stack_line_numbers(result_json["exec_stack"]["execution_stack"], target_path)

        # 清理 tracing 代码带来的额外项
        exec_stack = result_json.get("exec_stack", {})
        if "execution_stack" in exec_stack:
            filtered_stack = []
            for frame in exec_stack["execution_stack"]:
                details = frame.get("details", frame)
                module = details.get("module") or details.get("module_name")
                if not (module and str(module).startswith("VarTracer")):
                    filtered_stack.append(frame)
            exec_stack["execution_stack"] = filtered_stack

        dependency = result_json.get("dependency", {})
        for dep_file in list(dependency.keys()):
            var_items = dependency[dep_file]
            keys_to_del = [k for k, v in var_items.items() if v.get("variableName") == "exec_stack_json"]
            for k in keys_to_del:
                del var_items[k]
            if dep_file.endswith("VarTracer_Code.py"):
                del dependency[dep_file]
        keys_to_del = [k for k in dependency.keys() if k.endswith("VarTracer_Core.py")]
        for k in keys_to_del:
            del dependency[k]

        # 去除所有(VarTracerTemp)字符串
        result_json = remove_vartracertemp_str(result_json)

        if print:
            print(json.dumps(result_json, indent=4))
        return result_json

    finally:
        # 删除临时文件和结果文件
        # if os.path.exists(result_path):
        #     os.remove(result_path)
        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        pass

# def compare_exec_stack(exec_stack1, exec_stack2, output_path, output_name="compare_exec_stack_result.xlsx"):
#     """
#     比较两个 VarTracer.exec_stack_json 生成的 execution stack（JSON 格式），
#     输出两部分差异：
#       1. A 中有但 B 中没有的代码行（文件路径+行号）
#       2. B 中有但 A 中没有的代码行（文件路径+行号） 
#     并将结果保存为 xlsx 文件，包含两个 sheet，按文件路径分组，文件路径纵向合并单元格。
#     """
#     import xlsxwriter
#     import os

#     output_xlsx = os.path.join(output_path, output_name)

#     def extract_lines(exec_stack):
#         """
#         提取所有 (file_path, line_no, line_content) 元组，按文件路径分组。
#         返回: {file_path: set((line_no, line_content))}
#         """
#         lines_by_file = {}
#         stack = exec_stack.get("execution_stack", [])
#         def traverse(stack):
#             for event in stack:
#                 details = event.get("details", {})
#                 file_path = details.get("file_path")
#                 line_no = details.get("line_no")
#                 line_content = details.get("line_content")
#                 event_type = event.get("type", "").upper()
#                 if event_type == "LINE" and file_path and line_no:
#                     lines_by_file.setdefault(file_path, set()).add((str(line_no), line_content))
#                 # 递归 daughter_stack
#                 if "daughter_stack" in details:
#                     traverse(details["daughter_stack"])
#         traverse(stack)
#         return lines_by_file

#     lines_A = extract_lines(exec_stack1)
#     lines_B = extract_lines(exec_stack2)

#     # 1. A 中有但 B 中没有
#     only_in_A = {}
#     for file_path, lines in lines_A.items():
#         diff = lines - lines_B.get(file_path, set())
#         if diff:
#             only_in_A[file_path] = diff

#     # 2. B 中有但 A 中没有
#     only_in_B = {}
#     for file_path, lines in lines_B.items():
#         diff = lines - lines_A.get(file_path, set())
#         if diff:
#             only_in_B[file_path] = diff

#     # 写入 xlsx
#     workbook = xlsxwriter.Workbook(output_xlsx)
#     border_fmt = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
#     header_fmt = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

#     def write_sheet(sheet_name, data):
#         worksheet = workbook.add_worksheet(sheet_name)
#         worksheet.write(0, 0, "File Path", header_fmt)
#         worksheet.write(0, 1, "Line No", header_fmt)
#         worksheet.write(0, 2, "Line Content", header_fmt)
#         row = 1
#         for file_path in sorted(data.keys()):
#             lines = sorted(list(data[file_path]), key=lambda x: int(x[0]))
#             n = len(lines)
#             if n == 0:
#                 continue
#             worksheet.merge_range(row, 0, row + n - 1, 0, file_path, border_fmt)
#             for i, (line_no, line_content) in enumerate(lines):
#                 worksheet.write(row + i, 1, line_no, border_fmt)
#                 worksheet.write(row + i, 2, line_content, border_fmt)
#             row += n

#     write_sheet("Only_in_A", only_in_A)
#     write_sheet("Only_in_B", only_in_B)
#     workbook.close()
#     return output_xlsx



def compare_exec_stack(exec_stack1, exec_stack2, output_path, output_name="compare_exec_stack_result.xlsx"):
    """
    比较两个 VarTracer.exec_stack_json 生成的 execution stack（JSON 格式），
    为每个事件赋予 event.no（顺序编号），并添加 event.endemic（只在当前 stack 中出现为 True）。
    生成的 xlsx 文件包含两个 sheet，分别为两个 execution stack，包含所有事件及上述属性。
    """
    import xlsxwriter
    import os

    output_xlsx = os.path.join(output_path, output_name)

    # 提取所有事件，展平成一维列表，保留顺序
    def flatten_stack(exec_stack):
        events = []
        def traverse(stack, no_counter):
            for event in stack:
                events.append((no_counter[0], event))
                no_counter[0] += 1
                details = event.get("details", {})
                if "daughter_stack" in details:
                    traverse(details["daughter_stack"], no_counter)
        traverse(exec_stack.get("execution_stack", []), [1])
        return events

    # 判断事件是否相同（文件路径、行号、行内容都相同）
    def event_key(event):
        details = event.get("details", {})
        return (
            details.get("file_path"),
            str(details.get("line_no")),
            str(details.get("line_content"))
        )

    # 展平两个 execution stack
    events1 = flatten_stack(exec_stack1)
    events2 = flatten_stack(exec_stack2)

    # 构建集合用于对比
    set2 = set(event_key(e) for _, e in events2)
    set1 = set(event_key(e) for _, e in events1)

    # 为每个事件赋予 event.no 和 event.endemic
    def annotate_events(events, other_set):
        annotated = []
        for no, event in events:
            key = event_key(event)
            event = dict(event)  # shallow copy
            event['no'] = no
            event['endemic'] = key not in other_set
            annotated.append(event)
        return annotated

    annotated1 = annotate_events(events1, set2)
    annotated2 = annotate_events(events2, set1)

    # 写入 xlsx
    workbook = xlsxwriter.Workbook(output_xlsx)
    border_fmt = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
    header_fmt = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    endemic_false_fmt = workbook.add_format({'font_color': '#C0C0C0'})  # 灰色字体

    headers = [
        "event.no", "file_path", "event.type", "module", "line_no", "line_content", "event.endemic"
    ]

    def write_sheet(sheet_name, data):
        worksheet = workbook.add_worksheet(sheet_name)
        for col, h in enumerate(headers):
            worksheet.write(0, col, h, header_fmt)
        worksheet.autofilter(0, 0, len(data), len(headers) - 1)
        for row_idx, event in enumerate(sorted(data, key=lambda x: x['no']), 1):
            details = event.get("details", {})
            row = [
                event['no'],
                details.get("file_path", ""),
                event.get("type", ""),
                details.get("module", ""),
                details.get("line_no", ""),
                details.get("line_content", ""),
                event['endemic']
            ]
            endemic = event['endemic']
            for col_idx, value in enumerate(row):
                if col_idx == 6 and not endemic:
                    worksheet.write(row_idx, col_idx, value, endemic_false_fmt)
                else:
                    worksheet.write(row_idx, col_idx, value, border_fmt)

    write_sheet("ExecStack1", annotated1)
    write_sheet("ExecStack2", annotated2)
    workbook.close()
    return output_xlsx

def compare_dependency(dep1, dep2, output_path, output_name="compare_dependency_result.xlsx"):
    """
    比较两个 VarTracer.dep_tree_json 生成的依赖树（JSON 格式），
    输出两部分差异：
      1. 只在 A 中有但 B 中没有的文件/变量
      2. 只在 B 中有但 A 中没有的文件/变量
    并将结果保存为 xlsx 文件，包含两个 sheet，按文件路径、变量名、变量行号纵向合并单元格。
    """
    import xlsxwriter
    import os

    output_xlsx = os.path.join(output_path, output_name)

    def extract_vars(dep):
        """
        提取所有 (file_path, var_name, line_no, results) 元组。
        返回: {(file_path, var_name, line_no): results}
        """
        result = {}
        for file_path, var_dict in dep.items():
            if file_path == "trace_started_at":
                continue
            for var_key, var_info in var_dict.items():
                var_name = var_info.get("variableName", var_key)
                line_no = str(var_info.get("lineNumber", "-"))
                results = var_info.get("results", {})
                result[(file_path, var_name, line_no)] = results
        return result

    vars_A = extract_vars(dep1)
    vars_B = extract_vars(dep2)

    # 1. 只在 A 中有但 B 中没有
    only_in_A = {k: v for k, v in vars_A.items() if k not in vars_B}
    # 2. 只在 B 中有但 A 中没有
    only_in_B = {k: v for k, v in vars_B.items() if k not in vars_A}

    def prepare_sheet_data(var_dict):
        """
        生成 sheet 数据，每一行是
        (file_path, var_name, line_no, dep_var, first_occurrence, co_occurrences)
        """
        rows = []
        for (file_path, var_name, line_no), results in var_dict.items():
            if not results:
                # 没有依赖变量
                rows.append((file_path, var_name, line_no, "-", "-", "-"))
            else:
                for dep_var, dep_info in results.items():
                    first_occ = str(dep_info.get("first_occurrence", "-"))
                    co_occs = dep_info.get("co_occurrences", [])
                    if isinstance(co_occs, list):
                        co_occs_str = ", ".join(str(x) for x in co_occs) if co_occs else "-"
                    else:
                        co_occs_str = "-"
                    rows.append((file_path, var_name, line_no, dep_var, first_occ, co_occs_str))
        return rows

    data_A = prepare_sheet_data(only_in_A)
    data_B = prepare_sheet_data(only_in_B)

    # 写入 xlsx
    workbook = xlsxwriter.Workbook(output_xlsx)
    border_fmt = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
    header_fmt = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

    headers = ["File Path", "Variable", "Line No", "Dep.Var", "First Occurrence", "Co-occurrences"]

    def write_sheet(sheet_name, data):
        worksheet = workbook.add_worksheet(sheet_name)
        for col, h in enumerate(headers):
            worksheet.write(0, col, h, header_fmt)
        if not data:
            return

        from collections import defaultdict
        # 先按文件分组，再按变量名+行号分组
        file_group = defaultdict(list)
        for row in data:
            file_group[row[0]].append(row)

        row_idx = 1
        for file_path in sorted(file_group.keys()):
            file_rows = file_group[file_path]
            # 按变量名+行号分组
            var_group = defaultdict(list)
            for row in file_rows:
                var_group[(row[1], row[2])].append(row)
            var_items = sorted(var_group.items(), key=lambda x: (x[0][0], int(x[0][1]) if str(x[0][1]).isdigit() else 0))
            file_row_start = row_idx
            for (var_name, line_no), rows in var_items:
                n = len(rows)
                # 合并变量名和行号
                worksheet.merge_range(row_idx, 1, row_idx + n - 1, 1, var_name, border_fmt)
                worksheet.merge_range(row_idx, 2, row_idx + n - 1, 2, line_no, border_fmt)
                for i, row in enumerate(rows):
                    worksheet.write(row_idx + i, 3, row[3], border_fmt)
                    worksheet.write(row_idx + i, 4, row[4], border_fmt)
                    worksheet.write(row_idx + i, 5, row[5], border_fmt)
                row_idx += n
            # 合并文件名
            worksheet.merge_range(file_row_start, 0, row_idx - 1, 0, file_path, border_fmt)

    write_sheet("Only_in_A", data_A)
    write_sheet("Only_in_B", data_B)
    workbook.close()
    return output_xlsx
