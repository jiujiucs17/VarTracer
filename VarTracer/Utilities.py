import ast
import os
import subprocess
import json
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []
from datetime import datetime
# import openpyxl
# from openpyxl.comments import Comment
import re

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

def _dep_tree_symbol_records(dep_tree):
    files = dep_tree.get("files", {})
    records = {}
    for symbol_id, symbol_meta in dep_tree.get("symbols", {}).items():
        if len(symbol_meta) < 5:
            continue
        kind, name, scope, file_id, line_no = symbol_meta
        records[symbol_id] = {
            "kind": kind,
            "name": name,
            "scope": scope,
            "file_id": file_id,
            "file_path": files.get(file_id, ""),
            "line": line_no,
        }
    return records

def _dep_tree_incoming_edges(dep_tree):
    incoming = {}
    for edge in dep_tree.get("edges", []):
        if len(edge) < 5:
            continue
        src, dst, kind, line_no, hits = edge
        incoming.setdefault(dst, []).append({
            "src": src,
            "kind": kind,
            "line": line_no,
            "hits": hits,
        })
    return incoming

def _dep_tree_symbol_label(record):
    return f"{record['scope']}::{record['name']}"


def _compact_json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dep_tree_to_edgelist_text(dep_tree):
    symbols = dep_tree.get("symbols", {})
    files = dep_tree.get("files", {})
    lines = [
        "# Read as: META metadata; FIL file; SYM symbol; EDG edge; PTH path.",
        (
            "# Key map: "
            "v=version,ts=trace_started_at,"
            "id=record id,p=path,bn=file basename,"
            "k=kind,n=name,s=scope,f=file_id,l=line,"
            "src=source_symbol_id,dst=target_symbol_id,h=hits,"
            "sink=sink_symbol_id,seq=ordered symbol ids in one dependency path"
        ),
        "META " + _compact_json_text({
            "v": dep_tree.get("version"),
            "ts": dep_tree.get("trace_started_at"),
            "file_count": len(files),
            "symbol_count": len(symbols),
            "edge_count": len(dep_tree.get("edges", [])),
            "sink_count": len(dep_tree.get("paths", {})),
        }),
    ]

    for file_id, path in files.items():
        lines.append("FIL " + _compact_json_text({
            "id": file_id,
            "p": path,
            "bn": os.path.basename(path) if path else "",
        }))

    for symbol_id, symbol_meta in symbols.items():
        if len(symbol_meta) < 5:
            continue
        kind, name, scope, file_id, line_no = symbol_meta
        lines.append("SYM " + _compact_json_text({
            "id": symbol_id,
            "k": kind,
            "n": name,
            "s": scope,
            "f": file_id,
            "l": line_no,
        }))

    for edge in dep_tree.get("edges", []):
        if len(edge) < 5:
            continue
        src, dst, kind, line_no, hits = edge
        lines.append("EDG " + _compact_json_text({
            "src": src,
            "dst": dst,
            "k": kind,
            "l": line_no,
            "h": hits,
        }))

    for sink_id, sink_paths in dep_tree.get("paths", {}).items():
        for path in sink_paths:
            lines.append("PTH " + _compact_json_text({
                "sink": sink_id,
                "seq": path,
            }))

    return "\n".join(lines)

class FrameSnapshot:
    __slots__ = ("file_name", "function_name", "line_no", "locals", "globals", "module_name")

    def __init__(self, frame, capture_scope_names=False):
        self.file_name = os.path.abspath(frame.f_code.co_filename)
        self.function_name = frame.f_code.co_name
        self.line_no = frame.f_lineno
        if capture_scope_names:
            self.locals = self._snapshot_local_names(frame)
            self.globals = self._snapshot_global_names(frame)
        else:
            self.locals = ()
            self.globals = ()
        self.module_name = frame.f_globals.get('__name__', None) 

    def _snapshot_local_names(self, frame):
        return list(frame.f_locals.keys())

    def _snapshot_global_names(self, frame):
        return [
            name
            for name in frame.f_code.co_names
            if name in frame.f_globals
        ]

    def to_dict(self):
        """Convert the FrameSnapshot to a dictionary for easier serialization."""
        return {
            "filename": self.file_name,
            "function_name": self.function_name,
            "line_no": self.line_no,
            "locals": self.locals,
            "globals": self.globals,
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
                "dep_tree = DependencyTree(call_stack=exec_stack_json)\n",
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
            files = dependency.get("files", {})
            target_file_ids = {file_id for file_id, path in files.items() if path == target_path}
            if not target_file_ids:
                return

            for symbol_meta in dependency.get("symbols", {}).values():
                if len(symbol_meta) < 5:
                    continue
                if symbol_meta[3] in target_file_ids and isinstance(symbol_meta[4], int):
                    symbol_meta[4] = symbol_meta[4] - 4

            symbols = dependency.get("symbols", {})
            for edge in dependency.get("edges", []):
                if len(edge) < 5:
                    continue
                dst_symbol = symbols.get(edge[1])
                if dst_symbol and len(dst_symbol) >= 5 and dst_symbol[3] in target_file_ids and isinstance(edge[3], int):
                    edge[3] = edge[3] - 4

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
        if dependency:
            symbol_records = _dep_tree_symbol_records(dependency)
            files = dependency.get("files", {})
            file_ids_to_remove = {
                file_id
                for file_id, path in files.items()
                if path.endswith("VarTracer_Code.py") or path.endswith("VarTracer_Core.py")
            }
            symbol_ids_to_remove = {
                symbol_id
                for symbol_id, record in symbol_records.items()
                if record["name"] == "exec_stack_json" or record["file_id"] in file_ids_to_remove
            }

            if symbol_ids_to_remove:
                dependency["symbols"] = {
                    symbol_id: symbol_meta
                    for symbol_id, symbol_meta in dependency.get("symbols", {}).items()
                    if symbol_id not in symbol_ids_to_remove
                }
                dependency["edges"] = [
                    edge
                    for edge in dependency.get("edges", [])
                    if edge[0] not in symbol_ids_to_remove and edge[1] not in symbol_ids_to_remove
                ]
                dependency["paths"] = {
                    sink_id: [
                        path for path in sink_paths
                        if sink_id not in symbol_ids_to_remove and all(node_id not in symbol_ids_to_remove for node_id in path)
                    ]
                    for sink_id, sink_paths in dependency.get("paths", {}).items()
                    if sink_id not in symbol_ids_to_remove
                }

            used_file_ids = {
                symbol_meta[3]
                for symbol_meta in dependency.get("symbols", {}).values()
                if len(symbol_meta) >= 5
            }
            dependency["files"] = {
                file_id: path
                for file_id, path in dependency.get("files", {}).items()
                if file_id in used_file_ids and file_id not in file_ids_to_remove
            }

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

def break_down_granularity(exec_stack, dep_tree):
    """
    将 exec_stack（JSON对象）展开为事件列表，并为每个事件添加 module_event 和 function_event 字段。
    每个事件包含如下字段：
      event_no, file_path, event_type, module, call_depth, line_no, line_content, variable, dep. variable,
      module_event, function_event
    """
    symbol_records = _dep_tree_symbol_records(dep_tree)
    incoming_edges = _dep_tree_incoming_edges(dep_tree)
    file_to_id = {path: file_id for file_id, path in dep_tree.get("files", {}).items()}

    def get_dep_variables(dep_tree, event):
        assigned_vars = event.get("details", {}).get("assigned_vars", [])
        event_file_path = event.get("details", {}).get("file_path", "")
        event_func = event.get("details", {}).get("func", "")
        event_scope = event_func or "<module>"
        event_line = event.get("details", {}).get("line_no", "")
        event_file_id = file_to_id.get(event_file_path)
        text = ""
        if not assigned_vars or len(assigned_vars) == 0:
            return ""
        else:
            for var in assigned_vars:
                target_symbol_id = None
                for symbol_id, record in symbol_records.items():
                    if (
                        record["name"] == var
                        and record["scope"] == event_scope
                        and record["file_id"] == event_file_id
                        and str(record["line"]) == str(event_line)
                    ):
                        target_symbol_id = symbol_id
                        break

                dependent_vars = []
                for edge in incoming_edges.get(target_symbol_id, []):
                    record = symbol_records.get(edge["src"])
                    if record:
                        dependent_vars.append(record["name"])

                text_dep_vars = f"{var}: DEPENDENCIES: "
                text_dep_vars += f"{', '.join(sorted(set(dependent_vars)))}" if dependent_vars else "NONE"
                text += text_dep_vars + " ｜-|-|-｜ "
            
            return text.strip(" ｜-|-|-｜ ")

    # 展平 execution_stack
    events = []
    def traverse(stack, event_no_counter):
        for event in stack:
            details = event.get("details", {})
            # 变量名
            assigned_vars = details.get("assigned_vars", [])
            variable = assigned_vars[0] if assigned_vars else ""
            # 依赖变量
            dep_variable = get_dep_variables(dep_tree, event)

            # if module name is "(unknown-module)", use string "top level script" instead
            if details.get("module") == "(unknown-module)":
                module_name = "**** top level script ****"
            else:
                module_name = details.get("module")
            events.append({
                "event_no": event_no_counter[0],
                "file_path": details.get("file_path", ""),
                "event_type": event.get("type", ""),
                "module": module_name,
                "function": details.get("func", ""),
                "call_depth": details.get("depth", ""),
                "line_no": details.get("line_no", ""),
                "line_content": details.get("line_content", ""),
                "variable": variable,
                "dep.variable": dep_variable,
                # module_event/function_event 暂时空
            })
            event_no_counter[0] += 1
            # 递归 daughter_stack
            if "daughter_stack" in details:
                traverse(details["daughter_stack"], event_no_counter)
    traverse(exec_stack.get("execution_stack", []), [1])

    # 计算 module_event 和 function_event
    module_count = {}
    function_count = {}
    last_module = None
    last_function = None
    module_event_idx = 0
    function_event_idx = 0

    for idx, event in enumerate(events):
        module = event["module"]
        function = event.get("function")

        # module_event
        if module != last_module:
            module_event_idx = module_count.get(module, 0) + 1
            module_count[module] = module_event_idx
            last_module = module
        event["module_event"] = f"{module}_{module_event_idx}"

        # function_event
        func_key = (f"{module}_{module_event_idx}", function)
        if func_key != last_function:
            function_event_idx = function_count.get(func_key, 0) + 1
            function_count[func_key] = function_event_idx
            last_function = func_key
        event["function_event"] = f"{module}_{module_event_idx}_{function}_{function_event_idx}"

    return events

def compare_event_lists(exec_stack0, exec_stack1, dep_tree0, dep_tree1):
    """
    对比两个 exec_stack，分别展开为事件列表，并为每个事件添加 unique_to_this_feature 字段。
    如果某事件仅在 events1 或 events2 中出现，则其 unique_to_this_feature 字段为 True。
    判断标准为 (file_path, line_no, line_content) 三元组完全一致。
    返回 (events1, events2)
    """
    events0 = break_down_granularity(exec_stack0, dep_tree0)
    events1 = break_down_granularity(exec_stack1, dep_tree1)

    def make_event_key(event):
        return (event.get("file_path", ""), str(event.get("line_no", "")), str(event.get("line_content", "")))

    set1 = set(make_event_key(e) for e in events1)
    set0 = set(make_event_key(e) for e in events0)

    for e in events0:
        e["unique_to_this_feature"] = make_event_key(e) not in set1
    for e in events1:
        e["unique_to_this_feature"] = make_event_key(e) not in set0

    return events0, events1

def compare_exec_stack(exec_stack0, exec_stack1, dep_tree0, dep_tree1, output_path):
    """
    对比两个 exec_stack，分别以 module granularity、function granularity、event granularity 展示。
    输出为 xlsx 文件，包含多个 sheet：
      1. module_granularity：每个 module_event 的信息
      2. module_event_n：每个 module_event 的 func_event 信息，包含所有 func_event，并有 this_context 字段
      3. module_event_n_func_event_m：每个 func_event 的 line_event 信息，包含所有 line_event，并有 this_context 字段
    所有 sheet 都包含完整数据，并添加 this_context 字段。
    保存后用 openpyxl 设置除 module_granularity 外所有 sheet 的筛选条件为 this_context=true。
    """

    # 1. 生成 events0, events1
    events0, events1 = compare_event_lists(exec_stack0, exec_stack1, dep_tree0, dep_tree1)

    def build_json(events):
        # 提取所有 func_event
        func_event_map = {}
        func_event_list = []
        for e in events:
            fe = e["function_event"]
            if fe not in func_event_map:
                func_event_map[fe] = []
            func_event_map[fe].append(e)

        for fe, fe_events in func_event_map.items():
            func_name = fe_events[0].get("function_name") or fe_events[0].get("function") or ""
            file_path = fe_events[0].get("file_path", "")
            fe_event_nos = [ev["event_no"] for ev in fe_events]
            fe_unique_to_this_feature = any(ev["unique_to_this_feature"] for ev in fe_events)
            line_events = []
            for lev in fe_events:
                line_events.append({
                    "event_no": lev["event_no"],
                    "event_type": lev["event_type"],
                    "call_depth": lev["call_depth"],
                    "file_path": lev["file_path"],
                    "line_no": lev["line_no"],
                    "line_content": lev["line_content"],
                    "variable": lev["variable"],
                    "dep.variable": lev["dep.variable"],
                    "unique_to_this_feature": lev["unique_to_this_feature"],
                    "module_event": lev["module_event"],
                    "function_event": fe
                })
            func_event_list.append({
                "func_event_no": len(func_event_list) + 1,
                "func_event_identifier": fe, #fe_events[0]["function_event"],
                "func_name": func_name,
                "file_path": file_path,
                "first_event_no": min(fe_event_nos),
                "last_event_no": max(fe_event_nos),
                "unique_to_this_feature": fe_unique_to_this_feature,
                "line_events": line_events,
                "module_event": fe_events[0]["module_event"]
            })
        # 提取所有 module_event
        module_event_map = {}
        module_event_list = []
        for e in events:
            me = e["module_event"]
            if me not in module_event_map:
                module_event_map[me] = []
            module_event_map[me].append(e)
        for idx, (me, me_events) in enumerate(module_event_map.items(), 1):
            module_name = me_events[0]["module"]
            event_nos = [ev["event_no"] for ev in me_events]
            unique_to_this_feature = any(ev["unique_to_this_feature"] for ev in me_events)
            module_event_list.append({
                "module_event_no": idx,
                "module_event_identifier": me,
                "module_name": module_name,
                "first_event_no": min(event_nos),
                "last_event_no": max(event_nos),
                "unique_to_this_feature": unique_to_this_feature
            })
        return module_event_list, func_event_list

    def write_xlsx(events, filename):
        module_event_list, func_event_list = build_json(events)

        def write_headers_with_comments(worksheet, headers):
            for col, h in enumerate(headers):
                worksheet.write(0, col, h)
                clean_h = re.sub(r"\d+", "", h)
                if clean_h in label_meanings:
                    worksheet.write_comment(0, col, label_meanings[clean_h], {'width': 200, 'height': 100})
        # 提取所有 line_event
        all_line_events = []
        for func in func_event_list:
            for le in func["line_events"]:
                all_line_events.append(le)
        workbook = xlsxwriter.Workbook(os.path.join(output_path, filename))
        highlight_cell_format_pale = workbook.add_format({
            'bg_color': "#FFEDEF",    # 背景色
            'font_color': '#9C0006',  # 字体颜色
            'border': 1,  # 边框
            'align': 'left', 
            'valign': 'vcenter',
            'align': 'center'
        })
        highlight_cell_format_shine = workbook.add_format({
            'bg_color': "#FFC7CE",    # 背景色
            'font_color': '#9C0006',  # 字体颜色
            'border': 1,  # 边框
            'align': 'left', 
            'valign': 'vcenter',
            'align': 'center'

        })
        regular_cell_format = workbook.add_format({
            'bg_color': '#FFFFFF',    # 背景色
            'font_color': '#000000',  # 字体颜色
            'border': 1,  # 边框
            'align': 'left', 
            'valign': 'vcenter',
            'align': 'center'
        })
        hyperlink_format = workbook.add_format({
            'font_color': '#0000FF',  # 超链接字体颜色
            'underline': 1,           # 下划线
            'align': 'left', 
            'valign': 'vcenter',
            'align': 'left'
        })
        # 1. module_granularity sheet
        module_headers = [
            "module_event_no", "module_name", "execution_sequence_slice", "unique_to_this_feature", "associated_func_events" 
        ]
        module_sheet = workbook.add_worksheet("module_granularity")
        write_headers_with_comments(module_sheet, module_headers)
        module_sheet.autofilter(0, 0, len(module_event_list), len(module_headers) - 1)

        for row, module in enumerate(module_event_list, 1):
            mrow_format = regular_cell_format
            if module["unique_to_this_feature"]:
                mrow_format = highlight_cell_format_shine

            for col, key in enumerate(module_headers[:-1]):  # 最后一个字段 "associated_func_events" 不在 module_event_list 中
                if key == "execution_sequence_slice":
                    module_sheet.write(row, col, f"No. {module['first_event_no']} to {module['last_event_no']} of all exec events", mrow_format)
                else:
                    module_sheet.write(row, col, module[key], mrow_format)

        # 2. 每个 module_event_n sheet
        for row, module in tqdm(enumerate(module_event_list, 1), desc="Generating module_events & func_events", total=len(module_event_list)):
            func_headers = [
            "func_event_no", "func_name", "file_path", "module_event_no", "execution_sequence_slice", "unique_to_this_feature", f"within_module_event_{row}", "associated_line_events"
        ]

            sheet_name = f"m_evt_{module['module_event_no']}"
            func_sheet = workbook.add_worksheet(sheet_name)

            if len(func_event_list) == 0:
                # 如果没有 func_event，则跳过这个 module_event 的 sheet
                module_sheet.write(row, len(module_headers) - 1, "-")
                continue

            module_sheet.write_url(row, len(module_headers) - 1, 
                                   f"internal:'{sheet_name}'!A1",
                                   string=f"func_events")  # 添加超链接到 func_event sheet

            write_headers_with_comments(func_sheet, func_headers)
            func_sheet.autofilter(0, 0, len(func_event_list), len(func_headers) - 1)
            
            for frow, func in enumerate(func_event_list, 1):
                this_context = func["module_event"] == module["module_event_identifier"]
                if this_context and func["unique_to_this_feature"]:
                    frow_format = highlight_cell_format_shine
                elif not this_context and func["unique_to_this_feature"]:
                    frow_format = highlight_cell_format_pale
                else:
                    frow_format = regular_cell_format

                for col, key in enumerate(func_headers[:-1]):  # 最后一个字段 "associated_line_events" 不在 func_event_list 中
                    if key == f"within_module_event_{row}":
                        func_sheet.write(frow, col, this_context, frow_format)
                    elif key == "execution_sequence_slice":
                        func_sheet.write(frow, col, f"No. {func['first_event_no']} to {func['last_event_no']} of all exec events", frow_format)
                    elif key == "module_event_no":
                        # 查询 identifier为 module_event 的 module_event_no
                        module_event_no = next((me["module_event_no"] for me in module_event_list if me["module_event_identifier"] == func["module_event"]), None)
                        func_sheet.write(frow, col, module_event_no, frow_format)
                    else:
                        func_sheet.write(frow, col, func[key] if key in func else "", frow_format)

                # 3. 每个 module_event_n_func_event_m sheet
                if this_context:
                    event_headers = [
                        "event_no", "event_type", "call_depth", "file_path", "module_event_no", "function_event_no", "line_no",
                        "line_content", "variable", "dep.variable", "unique_to_this_feature", f"within_func_event_{frow}"
                    ]
                    event_sheet_name = f"m_evt_{module['module_event_no']}_f_evt_{frow}"
                    event_sheet = workbook.add_worksheet(event_sheet_name)

                    func_sheet.write_url(frow, len(func_headers) - 1,
                                         f"internal:'{event_sheet_name}'!A1",
                                         string=f"line_events")

                    write_headers_with_comments(event_sheet, event_headers)
                    event_sheet.autofilter(0, 0, len(all_line_events), len(event_headers) - 1)
                    for erow, event in enumerate(all_line_events, 1):
                        this_context_event = event["function_event"] == func["func_event_identifier"]
                        # erow_format = highlight_cell_format if this_context_event else regular_cell_format

                        if this_context_event and event["unique_to_this_feature"]:
                            erow_format = highlight_cell_format_shine
                        elif not this_context_event and event["unique_to_this_feature"]:
                            erow_format = highlight_cell_format_pale
                        else:
                            erow_format = regular_cell_format

                        for ecol, eh in enumerate(event_headers):
                            if eh == f"within_func_event_{frow}":
                                event_sheet.write(erow, ecol, this_context_event, erow_format)
                            elif eh == "module_event_no":
                                # 查询 identifier为 module_event 的 module_event_no
                                module_event_no = next((me["module_event_no"] for me in module_event_list if me["module_event_identifier"] == event["module_event"]), None)
                                event_sheet.write(erow, ecol, module_event_no, erow_format)
                            elif eh == "function_event_no":
                                # 查询 identifier为 func_event 的 func_event_no
                                func_event_no = next((fe["func_event_no"] for fe in func_event_list if fe["func_event_identifier"] == event["function_event"]), None)
                                event_sheet.write(erow, ecol, func_event_no, erow_format)
                            else:
                                value = "-"
                                if eh in event:
                                    value = event[eh]
                                    if value is None or value == "":
                                        value = "-"
                                event_sheet.write(erow, ecol, value, erow_format)

                    last_col = event_sheet.dim_colmax if hasattr(event_sheet, 'dim_colmax') and event_sheet.dim_colmax is not None else 0
                    last_row = event_sheet.dim_rowmax if hasattr(event_sheet, 'dim_rowmax') and event_sheet.dim_rowmax is not None else 0
                    # 在最后一列右侧第一个单元格添加超链接
                    event_sheet.write_url(0, last_col + 2, f"internal:'{sheet_name}'!A1", cell_format=hyperlink_format, string=f"back to function granularity")
                    event_sheet.write_url(last_row + 2, 0, f"internal:'{sheet_name}'!A1", cell_format=hyperlink_format, string=f"back to function granularity")
                else:
                    func_sheet.write(frow, len(func_headers) - 1, "-")
        
        print("cleaning up workbooks...")
        for worksheet in workbook.worksheets():
            # 设置冻结窗格
            worksheet.freeze_panes(1, 0)
            # 设置默认列宽
            default_width = 25
            worksheet.set_column('A:XFD', default_width)
            # 获取当前工作表的数据行数和列数
            last_row = worksheet.dim_rowmax if hasattr(worksheet, 'dim_rowmax') and worksheet.dim_rowmax is not None else 1
            last_col = worksheet.dim_colmax if hasattr(worksheet, 'dim_colmax') and worksheet.dim_colmax is not None else 0
            # 在最后一列右侧第一个单元格添加超链接
            worksheet.write_url(0, (last_col - 1 if len(worksheet.name) >= 22 else last_col + 1), "internal:'module_granularity'!A1", cell_format=hyperlink_format, string="back to module granularity")
            worksheet.write_url((last_row + 1 if len(worksheet.name) >= 22 else last_row + 2), 0, "internal:'module_granularity'!A1", cell_format=hyperlink_format, string="back to module granularity")

        workbook.close()

        # openpyxl logic removed for performance optimization
        pass



    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    write_xlsx(events0, f"stack0_{timestamp}.xlsx")
    write_xlsx(events1, f"stack1_{timestamp}.xlsx")

    print(f"length of events0: {len(events0)}")
    print(f"length of events1: {len(events1)}")

    return None, None

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
        提取所有 (file_path, var_name, line_no) 元组。
        返回: {(file_path, var_name, line_no): [dependency labels]}
        """
        result = {}
        symbol_records = _dep_tree_symbol_records(dep)
        incoming_edges = _dep_tree_incoming_edges(dep)
        for symbol_id, record in symbol_records.items():
            if record["kind"] in {"param", "ref"}:
                continue
            dependencies = []
            for edge in incoming_edges.get(symbol_id, []):
                source_record = symbol_records.get(edge["src"])
                if source_record:
                    dependencies.append(_dep_tree_symbol_label(source_record))
            result[(record["file_path"], record["name"], str(record["line"]))] = sorted(set(dependencies))
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
                for dep_var in results:
                    rows.append((file_path, var_name, line_no, dep_var, "-", "-"))
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


label_meanings = {
    "module_event_no": "Module Event number in the entire Execution Stack, representing the order of Module Events being executed.",
    "module_name": "Name of the module where the event occurred.",
    "execution_sequence_slice": "Slice of the execution sequence, indicating the range of Line Event numbers contained by this Event.",
    "unique_to_this_feature": "Indicates whether this Event only showed up in the script with this feature enabled.",
    "associated_func_events": "Hyperlink to the Function Event sheet, which shows all Function Events associated with this Module Event.",

    "func_event_no": "Function Event number in the entire Execution Stack, representing the order of Function Events being executed.",
    "func_name": "Name of the Function where the event occurred.",
    "file_path": "File path where the event occurred.",
    "within_module_event_": "Indicates whether this Function Event is within the context of the Module Event identified by the label.",
    "associated_line_events": "Hyperlink to the Line Event sheet, which shows all Line Events associated with this Function Event.",

    "event_no": "Event number in the entire Execution Stack, representing the order of Line Events being executed.",
    "event_type": "Type of the event, one of the following: 'LINE', 'CALL', 'RETURN', 'EXCEPTION'.",
    "call_depth": "Call depth of the event, indicating how deep the call stack is at this point.",
    "module_event_no": "Module Event number associated with this Line Event, conceptually linking it to the Module Event sheet.",
    "function_event_no": "Function Event number associated with this Line Event, conceptually linking it to the Function Event sheet.",
    "line_no": "Line number in the python file where the Line Event occurred.",
    "line_content": "Code content of the Line Event.",
    "variable": "Variable being modified or assigned value to with this Line Event, if any.",
    # considering changing the dep.variable meaning to "variables whose value **this line** depends on, blabla"
    "dep.variable": "Variables whose value the variable in the left depends on, showing inline dependencies and all dependencies within the file.",
    "within_func_event_": "Indicates whether this Line Event is within the context of the Function Event identified by the label.",

    "back to module granularity": "Hyperlink to the Module Granularity sheet, which shows all Module Events.",
    "back to function granularity": "Hyperlink to the Function Granularity sheet, which shows all Function Events. While highlighting the current context.",
}

def extract_unique_functions(exec_stack_1, exec_stack_0, output_folder, generate_llm_txt=False):
    """
    对比两个 execution trace，输出 trace A 相对于 trace B 的 unique modules/files/functions。

    参数约定：
    - exec_stack_1: execution trace A（feature positive）
    - exec_stack_0: execution trace B（feature negative）
    - output_path: 输出目录
    - generate_llm_txt: 默认为 False。为 True 时，额外生成一个压缩后的 LLM 文本文件

    兼容以下输入形式：
    - {"execution_stack": [...]}
    - {"exec_stack": {"execution_stack": [...]}}
    - 直接传入 execution_stack 列表

    输出文件：
    - {output_path}/unique_artifacts.json
    - {output_path}/unique_artifacts.txt（仅当 generate_llm_txt=True）

    输出 JSON 结构：
    {
        "comparison": {...},
        "unique_modules": [...],
        "unique_files": [...],
        "unique_functions": [...]
    }
    """

    def normalize_trace_stack(trace_obj):
        if isinstance(trace_obj, dict):
            if isinstance(trace_obj.get("execution_stack"), list):
                return trace_obj["execution_stack"]
            if isinstance(trace_obj.get("exec_stack"), dict):
                return normalize_trace_stack(trace_obj["exec_stack"])
        return trace_obj if isinstance(trace_obj, list) else []

    def safe_int(value):
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def get_relative_path(path):
        if not path:
            return ""
        try:
            return os.path.relpath(path, os.getcwd())
        except ValueError:
            return path

    def compact_bool(value):
        return 1 if value else 0

    def compact_samples(samples):
        packed = []
        for sample in samples:
            line_no = sample.get("line_number")
            line_content = str(sample.get("line_content", "")).replace("\n", "\\n")
            packed.append(f"{line_no}:{line_content}")
        return packed

    def build_llm_text(payload):
        comparison = payload.get("comparison", {})
        lines = [
            "# Read as: CMP comparison; MOD unique module; FIL unique file; FUN unique function.",
            (
                "# Key map: "
                "a=trace_a_role,b=trace_b_role,rel=comparison_type,"
                "um/uf/ufn=unique counts,"
                "n=name,module_name,q=qualified_name,top=is_top_level_script,"
                "p=path,rp=relative_path,bn=file_name,"
                "fps=file_paths,bns=file_names,mods=module_names,fns=function_names,"
                "fc/file_count,mc/module_count,fnc=function_count,"
                "def=defined_line_number,end=end_line_number,"
                "type=definition_type,cls=class_name,parent=parent_scope,"
                "async=is_async,src=definition_found_in_source,"
                "first=first_seen_line_number,obs=observed_line_numbers,"
                "ev=event_count,cal/call=line-call counts,ret=return_count,exc=exception_count,"
                "dep=max_call_depth,smp=sample_executed_lines"
            ),
            "CMP " + _compact_json_text({
                "a": comparison.get("trace_a_role"),
                "b": comparison.get("trace_b_role"),
                "rel": comparison.get("comparison_type"),
                "um": comparison.get("unique_module_count"),
                "uf": comparison.get("unique_file_count"),
                "ufn": comparison.get("unique_function_count"),
            }),
        ]

        for record in payload.get("unique_modules", []):
            lines.append("MOD " + _compact_json_text({
                "n": record.get("module_name"),
                "top": compact_bool(record.get("is_top_level_script")),
                "fps": record.get("file_paths", []),
                "bns": record.get("file_names", []),
                "fns": record.get("function_names", []),
                "fc": record.get("file_count"),
                "fnc": record.get("function_count"),
                "ev": record.get("event_count_in_trace_a"),
                "ln": record.get("line_event_count_in_trace_a"),
                "cal": record.get("call_event_count_in_trace_a"),
                "dep": record.get("max_call_depth"),
                "first": record.get("first_seen_line_number"),
            }))

        for record in payload.get("unique_files", []):
            lines.append("FIL " + _compact_json_text({
                "p": record.get("path"),
                "rp": record.get("relative_path"),
                "bn": record.get("file_name"),
                "mods": record.get("module_names", []),
                "fns": record.get("function_names", []),
                "mc": record.get("module_count"),
                "fnc": record.get("function_count"),
                "ev": record.get("event_count_in_trace_a"),
                "ln": record.get("line_event_count_in_trace_a"),
                "cal": record.get("call_event_count_in_trace_a"),
                "dep": record.get("max_call_depth"),
                "first": record.get("first_seen_line_number"),
            }))

        for record in payload.get("unique_functions", []):
            lines.append("FUN " + _compact_json_text({
                "n": record.get("name"),
                "q": record.get("qualified_name"),
                "mod": record.get("module_name"),
                "p": record.get("path"),
                "rp": record.get("relative_path"),
                "bn": record.get("file_name"),
                "def": record.get("defined_line_number"),
                "end": record.get("end_line_number"),
                "type": record.get("definition_type"),
                "cls": record.get("class_name"),
                "parent": record.get("parent_scope"),
                "async": compact_bool(record.get("is_async")),
                "src": compact_bool(record.get("definition_found_in_source")),
                "first": record.get("first_seen_line_number"),
                "obs": record.get("observed_line_numbers", []),
                "ev": record.get("event_count_in_trace_a"),
                "call": record.get("call_count_in_trace_a"),
                "line": record.get("line_event_count_in_trace_a"),
                "ret": record.get("return_count_in_trace_a"),
                "exc": record.get("exception_count_in_trace_a"),
                "dep": record.get("max_call_depth"),
                "smp": compact_samples(record.get("sample_executed_lines", [])),
            }))

        return "\n".join(lines)

    definition_cache = {}

    def parse_function_definitions(file_path):
        if not file_path:
            return []
        if file_path in definition_cache:
            return definition_cache[file_path]

        definitions = []
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                source = handle.read()
            tree = ast.parse(source, filename=file_path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            definition_cache[file_path] = definitions
            return definitions

        def walk_nodes(body, scope_parts=None, class_stack=None, parent_kind=None):
            scope_parts = list(scope_parts or [])
            class_stack = list(class_stack or [])

            for node in body:
                if isinstance(node, ast.ClassDef):
                    walk_nodes(
                        getattr(node, "body", []),
                        scope_parts=scope_parts + [node.name],
                        class_stack=class_stack + [node.name],
                        parent_kind="class",
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified_name = ".".join(scope_parts + [node.name]) if scope_parts else node.name
                    if parent_kind == "class":
                        definition_type = "method"
                    elif parent_kind == "function":
                        definition_type = "nested_function"
                    else:
                        definition_type = "function"

                    definitions.append({
                        "name": node.name,
                        "qualified_name": qualified_name,
                        "path": file_path,
                        "defined_line_number": safe_int(getattr(node, "lineno", None)),
                        "end_line_number": safe_int(getattr(node, "end_lineno", getattr(node, "lineno", None))),
                        "class_name": class_stack[-1] if class_stack else None,
                        "parent_scope": ".".join(scope_parts) if scope_parts else None,
                        "definition_type": definition_type,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    })
                    walk_nodes(
                        getattr(node, "body", []),
                        scope_parts=scope_parts + [node.name],
                        class_stack=class_stack,
                        parent_kind="function",
                    )

        walk_nodes(getattr(tree, "body", []))
        definition_cache[file_path] = definitions
        return definitions

    def resolve_function_definition(file_path, func_name, observed_lines):
        candidates = [
            item for item in parse_function_definitions(file_path)
            if item.get("name") == func_name
        ]
        if not candidates:
            return {}
        clean_lines = [line for line in observed_lines if isinstance(line, int)]
        if not clean_lines and len(candidates) == 1:
            return dict(candidates[0])

        def candidate_score(candidate):
            start = candidate.get("defined_line_number")
            end = candidate.get("end_line_number") or start
            hits = 0
            for line in clean_lines:
                if start is not None and end is not None and start <= line <= end:
                    hits += 1
            span = (end - start) if start is not None and end is not None else 10 ** 9
            start_rank = -(start or 10 ** 9)
            return (hits, -span, start_rank)

        return dict(max(candidates, key=candidate_score))

    def collect_frame_lines(events, target_file_path, target_func_name):
        lines = []
        for event in events:
            details = event.get("details", {})
            if (
                details.get("file_path") == target_file_path
                and details.get("func") == target_func_name
            ):
                line_no = safe_int(details.get("line_no"))
                if line_no is not None:
                    lines.append(line_no)
            if event.get("type") == "CALL":
                continue
        return lines

    def build_function_key(file_path, module_name, func_name, definition):
        defined_line = definition.get("defined_line_number")
        qualified_name = definition.get("qualified_name") or func_name
        return (file_path, module_name or "", qualified_name, defined_line if defined_line is not None else 0)

    def ensure_module_record(container, module_name):
        return container.setdefault(module_name, {
            "module_name": module_name,
            "is_top_level_script": module_name == "(unknown-module)",
            "file_paths": set(),
            "function_names": set(),
            "event_count_in_trace_a": 0,
            "line_event_count_in_trace_a": 0,
            "call_event_count_in_trace_a": 0,
            "max_call_depth": 0,
            "first_seen_line_number": None,
        })

    def ensure_file_record(container, file_path):
        return container.setdefault(file_path, {
            "path": file_path,
            "relative_path": get_relative_path(file_path),
            "file_name": os.path.basename(file_path) if file_path else "",
            "module_names": set(),
            "function_names": set(),
            "event_count_in_trace_a": 0,
            "line_event_count_in_trace_a": 0,
            "call_event_count_in_trace_a": 0,
            "max_call_depth": 0,
            "first_seen_line_number": None,
        })

    def ensure_function_record(container, function_key, base_info):
        return container.setdefault(function_key, {
            "name": base_info.get("name"),
            "qualified_name": base_info.get("qualified_name"),
            "module_name": base_info.get("module_name"),
            "path": base_info.get("path"),
            "relative_path": get_relative_path(base_info.get("path")),
            "file_name": os.path.basename(base_info.get("path")) if base_info.get("path") else "",
            "defined_line_number": base_info.get("defined_line_number"),
            "end_line_number": base_info.get("end_line_number"),
            "definition_type": base_info.get("definition_type"),
            "class_name": base_info.get("class_name"),
            "parent_scope": base_info.get("parent_scope"),
            "is_async": bool(base_info.get("is_async")),
            "definition_found_in_source": bool(base_info.get("definition_found_in_source")),
            "first_seen_line_number": None,
            "observed_line_numbers": set(),
            "event_count_in_trace_a": 0,
            "call_count_in_trace_a": 0,
            "line_event_count_in_trace_a": 0,
            "return_count_in_trace_a": 0,
            "exception_count_in_trace_a": 0,
            "max_call_depth": 0,
            "sample_executed_lines": [],
        })

    def prepare_function_identity(file_path, module_name, func_name, observed_lines):
        definition = resolve_function_definition(file_path, func_name, observed_lines)
        base_info = {
            "name": func_name,
            "qualified_name": definition.get("qualified_name") or func_name,
            "module_name": module_name,
            "path": file_path,
            "defined_line_number": definition.get("defined_line_number"),
            "end_line_number": definition.get("end_line_number"),
            "definition_type": definition.get("definition_type") or "function",
            "class_name": definition.get("class_name"),
            "parent_scope": definition.get("parent_scope"),
            "is_async": definition.get("is_async", False),
            "definition_found_in_source": bool(definition),
        }
        return build_function_key(file_path, module_name, func_name, definition), base_info

    def update_function_record(record, event, line_no, depth):
        event_type = event.get("type", "")
        record["event_count_in_trace_a"] += 1
        record["max_call_depth"] = max(record["max_call_depth"], depth or 0)
        if line_no is not None:
            record["observed_line_numbers"].add(line_no)
            if record["first_seen_line_number"] is None:
                record["first_seen_line_number"] = line_no
        if event_type == "CALL":
            record["call_count_in_trace_a"] += 1
        elif event_type == "LINE":
            record["line_event_count_in_trace_a"] += 1
        elif event_type == "RETURN":
            record["return_count_in_trace_a"] += 1
        elif event_type == "EXCEPTION":
            record["exception_count_in_trace_a"] += 1

        details = event.get("details", {})
        line_content = details.get("line_content")
        if line_no is not None and line_content:
            sample_key = (line_no, line_content)
            existing_keys = {
                (item.get("line_number"), item.get("line_content"))
                for item in record["sample_executed_lines"]
            }
            if sample_key not in existing_keys and len(record["sample_executed_lines"]) < 5:
                record["sample_executed_lines"].append({
                    "line_number": line_no,
                    "line_content": line_content,
                })

    def build_trace_summary(trace_obj):
        stack_list = normalize_trace_stack(trace_obj)
        state = {
            "modules": {},
            "files": {},
            "functions": {},
        }

        def traverse(events, active_function_key=None):
            for event in events:
                details = event.get("details", {})
                module_name = details.get("module")
                file_path = details.get("file_path")
                func_name = details.get("func")
                line_no = safe_int(details.get("line_no"))
                depth = safe_int(details.get("depth")) or 0
                event_type = event.get("type", "")

                if module_name:
                    module_record = ensure_module_record(state["modules"], module_name)
                    module_record["event_count_in_trace_a"] += 1
                    module_record["max_call_depth"] = max(module_record["max_call_depth"], depth)
                    if file_path:
                        module_record["file_paths"].add(file_path)
                    if func_name and func_name != "<module>":
                        module_record["function_names"].add(func_name)
                    if line_no is not None and module_record["first_seen_line_number"] is None:
                        module_record["first_seen_line_number"] = line_no
                    if event_type == "LINE":
                        module_record["line_event_count_in_trace_a"] += 1
                    elif event_type == "CALL":
                        module_record["call_event_count_in_trace_a"] += 1

                if file_path:
                    file_record = ensure_file_record(state["files"], file_path)
                    file_record["event_count_in_trace_a"] += 1
                    file_record["max_call_depth"] = max(file_record["max_call_depth"], depth)
                    if module_name:
                        file_record["module_names"].add(module_name)
                    if func_name and func_name != "<module>":
                        file_record["function_names"].add(func_name)
                    if line_no is not None and file_record["first_seen_line_number"] is None:
                        file_record["first_seen_line_number"] = line_no
                    if event_type == "LINE":
                        file_record["line_event_count_in_trace_a"] += 1
                    elif event_type == "CALL":
                        file_record["call_event_count_in_trace_a"] += 1

                current_function_key = active_function_key
                if func_name and func_name != "<module>" and file_path:
                    if event_type == "CALL":
                        observed_lines = collect_frame_lines(
                            details.get("daughter_stack", []),
                            file_path,
                            func_name,
                        )
                        current_function_key, base_info = prepare_function_identity(
                            file_path,
                            module_name,
                            func_name,
                            observed_lines,
                        )
                        function_record = ensure_function_record(
                            state["functions"],
                            current_function_key,
                            base_info,
                        )
                        update_function_record(function_record, event, line_no, depth)
                    else:
                        if current_function_key is None:
                            current_function_key, base_info = prepare_function_identity(
                                file_path,
                                module_name,
                                func_name,
                                [line_no] if line_no is not None else [],
                            )
                            function_record = ensure_function_record(
                                state["functions"],
                                current_function_key,
                                base_info,
                            )
                        else:
                            function_record = state["functions"][current_function_key]
                        update_function_record(function_record, event, line_no, depth)

                if event_type == "CALL":
                    traverse(details.get("daughter_stack", []), active_function_key=current_function_key)

        traverse(stack_list)

        for record in state["modules"].values():
            record["file_paths"] = sorted(record["file_paths"])
            record["file_names"] = sorted(os.path.basename(path) for path in record["file_paths"])
            record["function_names"] = sorted(record["function_names"])
            record["file_count"] = len(record["file_paths"])
            record["function_count"] = len(record["function_names"])

        for record in state["files"].values():
            record["module_names"] = sorted(record["module_names"])
            record["function_names"] = sorted(record["function_names"])
            record["module_count"] = len(record["module_names"])
            record["function_count"] = len(record["function_names"])

        for record in state["functions"].values():
            record["observed_line_numbers"] = sorted(record["observed_line_numbers"])
            record["sample_executed_lines"] = sorted(
                record["sample_executed_lines"],
                key=lambda item: (item.get("line_number") is None, item.get("line_number")),
            )

        return state

    trace_a_summary = build_trace_summary(exec_stack_1)
    trace_b_summary = build_trace_summary(exec_stack_0)

    unique_module_names = sorted(set(trace_a_summary["modules"]) - set(trace_b_summary["modules"]))
    unique_file_paths = sorted(set(trace_a_summary["files"]) - set(trace_b_summary["files"]))
    unique_function_keys = sorted(
        set(trace_a_summary["functions"]) - set(trace_b_summary["functions"]),
        key=lambda item: (item[0], item[3], item[2]),
    )
    unique_file_path_set = set(unique_file_paths)

    unique_module_records = []
    for module_name in unique_module_names:
        module_record = dict(trace_a_summary["modules"][module_name])
        module_file_paths = module_record.get("file_paths", [])
        filtered_file_paths = [
            file_path for file_path in module_file_paths
            if file_path in unique_file_path_set
        ]
        module_record["file_paths"] = filtered_file_paths
        module_record["file_names"] = [
            os.path.basename(file_path) for file_path in filtered_file_paths
        ]
        module_record["file_count"] = len(filtered_file_paths)
        unique_module_records.append(module_record)

    result_payload = {
        "comparison": {
            "trace_a_role": "feature_positive",
            "trace_b_role": "feature_negative",
            "comparison_type": "trace_a_minus_trace_b",
            "unique_module_count": len(unique_module_names),
            "unique_file_count": len(unique_file_paths),
            "unique_function_count": len(unique_function_keys),
        },
        "unique_modules": unique_module_records,
        "unique_files": [
            trace_a_summary["files"][file_path]
            for file_path in unique_file_paths
        ],
        "unique_functions": [
            trace_a_summary["functions"][function_key]
            for function_key in unique_function_keys
        ],
    }

    output_dir = output_folder or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "unique_artifacts.json")
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=4, ensure_ascii=False)

    llm_output_file = None
    if generate_llm_txt:
        llm_output_file = os.path.join(output_dir, "unique_artifacts.txt")
        with open(llm_output_file, "w", encoding="utf-8") as handle:
            handle.write(build_llm_text(result_payload))

    print(
        "\n\nSuccess! "
        f"modules={len(result_payload['unique_modules'])}, "
        f"files={len(result_payload['unique_files'])}, "
        f"functions={len(result_payload['unique_functions'])}\n\n"
        f"Unique trace summary extracted to {output_file}\n"
        + (
            f"LLM-ready text extracted to {llm_output_file}\n"
            if llm_output_file
            else ""
        )
    )
    return result_payload


def filter_dep_tree_by_unique_artifacts(unique_artifacts, dep_tree, output_folder, generate_llm_txt=False):
    """
    根据 extract_unique_functions() 生成的 unique_artifacts.json，对 feature-positive 的 dep_tree 做裁剪。

    参数：
    - unique_artifacts: unique_artifacts.json 的路径，或其已加载的 dict
    - dep_tree: feature-positive 对应 dep_tree json 的路径，或其已加载的 dict
    - output_folder: 输出目录
    - generate_llm_txt: 默认为 False。为 True 时，额外生成 edge-list 文本版本

    裁剪规则：
    1. 先根据 unique_artifacts 中出现的 module/file/function，在 dep_tree["paths"] 中找相关 path。
    2. 只保留命中的 path。
    3. 再根据被保留 path 中实际用到的 files/symbols/edges，反向精简 dep_tree 其余内容。

    输出文件：
    - {output_path}/unique_artifacts_dep_tree.json
    - {output_path}/unique_artifacts_dep_tree_edgelist.txt（仅当 generate_llm_txt=True）
    """

    def load_json_like(value, name):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            with open(value, "r", encoding="utf-8") as handle:
                return json.load(handle)
        raise TypeError(f"{name} must be a dict or a JSON file path.")

    def safe_int_local(value):
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def normalize_scope(value):
        return "" if value is None else str(value)

    def build_function_matchers(artifacts_json):
        matchers = []
        for record in artifacts_json.get("unique_functions", []):
            matchers.append({
                "path": record.get("path"),
                "name": record.get("name"),
                "defined_line_number": safe_int_local(record.get("defined_line_number")),
                "parent_scope": normalize_scope(record.get("parent_scope")),
                "class_name": normalize_scope(record.get("class_name")),
            })
        return matchers

    def symbol_matches_function(record, matcher):
        if matcher.get("path") and record.get("file_path") != matcher.get("path"):
            return False

        symbol_kind = record.get("kind")
        symbol_name = record.get("name")
        symbol_scope = normalize_scope(record.get("scope"))
        symbol_line = safe_int_local(record.get("line"))

        target_name = matcher.get("name")
        target_line = matcher.get("defined_line_number")
        parent_scope = matcher.get("parent_scope")
        class_scope = matcher.get("class_name")

        # 函数体内部的变量、参数、ref/attr 等符号通常都以函数名作为 scope。
        if symbol_scope == target_name:
            return True

        # 函数定义自身在 dep_tree 中通常表现为 kind=func，scope 为其父作用域（如 <module> 或类名）。
        if symbol_kind == "func" and symbol_name == target_name:
            if target_line is not None and symbol_line is not None and symbol_line != target_line:
                return False
            valid_parent_scopes = {parent_scope, class_scope}
            valid_parent_scopes.discard("")
            if not valid_parent_scopes:
                return symbol_scope in {"", "<module>"}
            return symbol_scope in valid_parent_scopes

        return False

    artifacts_json = load_json_like(unique_artifacts, "unique_artifacts")
    dep_tree_json = load_json_like(dep_tree, "dep_tree")
    output_dir = output_folder or os.getcwd()

    symbol_records = _dep_tree_symbol_records(dep_tree_json)
    dep_tree_paths = dep_tree_json.get("paths", {})

    target_file_paths = set()
    for record in artifacts_json.get("unique_modules", []):
        target_file_paths.update(record.get("file_paths", []))
    for record in artifacts_json.get("unique_files", []):
        if record.get("path"):
            target_file_paths.add(record["path"])

    function_matchers = build_function_matchers(artifacts_json)

    def symbol_matches_target(symbol_id):
        record = symbol_records.get(symbol_id)
        if not record:
            return False
        if record.get("file_path") in target_file_paths:
            return True
        return any(symbol_matches_function(record, matcher) for matcher in function_matchers)

    retained_paths = {}
    retained_symbol_ids = set()
    retained_edge_pairs = set()

    for sink_id, sink_paths in dep_tree_paths.items():
        matched_paths = []
        for path in sink_paths:
            if any(symbol_matches_target(symbol_id) for symbol_id in path):
                matched_paths.append(path)
                retained_symbol_ids.update(path)
                for index in range(len(path) - 1):
                    retained_edge_pairs.add((path[index], path[index + 1]))
        if matched_paths:
            retained_paths[sink_id] = matched_paths

    retained_edges = [
        edge for edge in dep_tree_json.get("edges", [])
        if len(edge) >= 2
        and edge[0] in retained_symbol_ids
        and edge[1] in retained_symbol_ids
        and (edge[0], edge[1]) in retained_edge_pairs
    ]

    retained_file_ids = {
        symbol_records[symbol_id]["file_id"]
        for symbol_id in retained_symbol_ids
        if symbol_id in symbol_records and symbol_records[symbol_id].get("file_id")
    }

    filtered_dep_tree = {}
    for key in ("_comment", "version", "trace_started_at"):
        if key in dep_tree_json:
            filtered_dep_tree[key] = dep_tree_json[key]

    filtered_dep_tree["files"] = {
        file_id: file_path
        for file_id, file_path in dep_tree_json.get("files", {}).items()
        if file_id in retained_file_ids
    }
    filtered_dep_tree["symbols"] = {
        symbol_id: symbol_meta
        for symbol_id, symbol_meta in dep_tree_json.get("symbols", {}).items()
        if symbol_id in retained_symbol_ids
    }
    filtered_dep_tree["edges"] = retained_edges
    filtered_dep_tree["paths"] = retained_paths
    filtered_dep_tree["filter_summary"] = {
        "source_unique_module_count": len(artifacts_json.get("unique_modules", [])),
        "source_unique_file_count": len(artifacts_json.get("unique_files", [])),
        "source_unique_function_count": len(artifacts_json.get("unique_functions", [])),
        "matched_path_count": sum(len(paths) for paths in retained_paths.values()),
        "matched_sink_count": len(retained_paths),
        "retained_file_count": len(filtered_dep_tree["files"]),
        "retained_symbol_count": len(filtered_dep_tree["symbols"]),
        "retained_edge_count": len(filtered_dep_tree["edges"]),
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "unique_dep_tree.json")
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(filtered_dep_tree, handle, indent=4, ensure_ascii=False)

    edgelist_output_file = None
    if generate_llm_txt:
        edgelist_output_file = os.path.join(output_dir, "unique_dep_tree_edgelist.txt")
        with open(edgelist_output_file, "w", encoding="utf-8") as handle:
            handle.write(_dep_tree_to_edgelist_text(filtered_dep_tree))

    print(
        "\n\nSuccess! "
        f"retained_paths={filtered_dep_tree['filter_summary']['matched_path_count']}, "
        f"retained_symbols={filtered_dep_tree['filter_summary']['retained_symbol_count']}, "
        f"retained_edges={filtered_dep_tree['filter_summary']['retained_edge_count']}\n\n"
        f"Filtered dep_tree extracted to {output_file}\n"
        + (
            f"Edge-list extracted to {edgelist_output_file}\n"
            if edgelist_output_file
            else ""
        )
    )
    return filtered_dep_tree
