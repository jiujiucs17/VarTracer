import os
import sys
import tempfile
import textwrap

from VarTracer.ASTParser import DependencyTree
from VarTracer.VarTracer_Core import VarTracer


class TraceTestMixin:
    """Shared helpers for tests that need to execute a temporary Python script."""

    def trace_source(self, source, clean_stdlib=True, extra_files=None):
        # This helper is the backbone of the integration tests:
        # it writes a tiny script to a temp file, runs it through VarTracer,
        # then returns every artifact the tests may want to inspect.
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "sample_script.py")
            normalized_source = textwrap.dedent(source).lstrip()
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(normalized_source)

            for relative_path, content in (extra_files or {}).items():
                extra_path = os.path.join(temp_dir, relative_path)
                os.makedirs(os.path.dirname(extra_path), exist_ok=True)
                with open(extra_path, "w", encoding="utf-8") as handle:
                    handle.write(textwrap.dedent(content).lstrip())

            with open(script_path, "r", encoding="utf-8") as handle:
                compiled = compile(handle.read(), script_path, "exec")

            vt = VarTracer(clean_stdlib=clean_stdlib, verbose=False)
            namespace = {"__name__": "__main__", "__file__": script_path}

            old_sys_path = list(sys.path)
            sys.path.insert(0, temp_dir)
            try:
                vt.start()
                exec(compiled, namespace)
            finally:
                vt.stop()
                sys.path[:] = old_sys_path

            exec_stack = vt.exec_stack_json(show_progress=False)
            parser = DependencyTree(call_stack=exec_stack)
            dep_tree = parser.parse_dependency()
            flow_trace = parser.parse_flow_trace()
            raw_result = vt.raw_result()

            return {
                "script_path": script_path,
                "exec_stack": exec_stack,
                "dep_tree": dep_tree,
                "flow_trace": flow_trace,
                "raw_result": raw_result,
                "vt": vt,
            }

    def flatten_events(self, events):
        # Execution trace assertions often need a flat list instead of the nested
        # daughter_stack structure, so this helper recursively flattens the tree.
        flattened = []
        for event in events:
            flattened.append(event)
            daughters = event.get("details", {}).get("daughter_stack", [])
            flattened.extend(self.flatten_events(daughters))
        return flattened

    def find_script_root_call(self, exec_stack, script_path):
        # The root CALL event for the temporary script is the anchor point for
        # nearly all execution-trace assertions.
        for event in exec_stack.get("execution_stack", []):
            details = event.get("details", {})
            if (
                event.get("type") == "CALL"
                and details.get("file_path") == script_path
                and details.get("func") == "<module>"
            ):
                return event
        self.fail(f"Expected to find root CALL event for script '{script_path}'")

    def find_var_info(self, file_deps, variable_name):
        # Dependency-graph tests usually care about the semantic variable name
        # rather than the compact symbol id.
        matches = []
        for symbol_id, symbol_meta in file_deps.get("symbols", {}).items():
            if len(symbol_meta) < 5:
                continue
            kind, name, scope, file_id, line_no = symbol_meta
            if name == variable_name:
                matches.append(
                    {
                        "id": symbol_id,
                        "kind": kind,
                        "name": name,
                        "scope": scope,
                        "file_id": file_id,
                        "line": line_no,
                    }
                )
        self.assertTrue(matches, f"Expected to find variable '{variable_name}' in dependency tree")
        return matches[0]

    def incoming_source_names(self, dep_tree, target_symbol_id):
        source_names = set()
        for edge in dep_tree.get("edges", []):
            if len(edge) < 5 or edge[1] != target_symbol_id:
                continue
            source_symbol = dep_tree.get("symbols", {}).get(edge[0], [])
            if len(source_symbol) >= 2:
                source_names.add(source_symbol[1])
        return source_names

    def incoming_edges(self, dep_tree, target_symbol_id):
        return [edge for edge in dep_tree.get("edges", []) if len(edge) >= 5 and edge[1] == target_symbol_id]
