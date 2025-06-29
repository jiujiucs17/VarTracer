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
