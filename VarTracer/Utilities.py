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
    
def extension_interface(file_path):
    """
    Modify a Python file to use VarTracer for dependency analysis, execute it, and restore the file.
    
    Args:
        file_path (str): Path to the Python file to be analyzed.
    
    Returns:
        dict: A dictionary containing the execution stack and dependency information.
    
    Raises:
        ValueError: If the file is not a .py file.
        FileNotFoundError: If the file does not exist.
    """
    # Check if the file exists and is a .py file
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.endswith('.py'):
        raise ValueError("The provided file is not a Python (.py) file.")
    
    # Get the directory of the target file
    file_dir = os.path.dirname(file_path)
    
    # Define paths for the backup file and result file in the same directory
    backup_path = os.path.join(file_dir, os.path.basename(file_path) + ".bak")
    result_path = os.path.join(file_dir, "result.json")
    
    # Backup the original file
    shutil.copy(file_path, backup_path)
    
    try:
        # Read the original file content
        with open(file_path, 'r') as f:
            original_content = f.readlines()
        
        # Add VarTracer imports and initialization at the top
        modified_content = ["from VarTracer import *\n", "import json\n", "vt = VarTracer()\n", "vt.start()\n"] + original_content
        
        # Add dependency analysis code at the end
        modified_content += [
            "\n",
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
        
        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.writelines(modified_content)
        
        # Execute the modified file in the current Python environment
        subprocess.run(['python3', file_path], check=True)
        
        # Read the result from the generated JSON file
        with open(result_path, 'r') as result_file:
            result_json = json.load(result_file)
        
        # print the result_json as a string
        print(json.dumps(result_json, indent=4))
        return result_json
    
    finally:
        # Restore the original file
        shutil.move(backup_path, file_path)
        # Clean up the temporary result file
        if os.path.exists(result_path):
            os.remove(result_path)