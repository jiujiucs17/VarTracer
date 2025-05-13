import inspect

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