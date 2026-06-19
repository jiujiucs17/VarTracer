import json
import os
import sys
import tempfile
import textwrap
import unittest

from VarTracer.VarTracer_Core import VarTracer
from VarTracer.test_code._trace_test_support import TraceTestMixin


class TestExecutionTrace(TraceTestMixin, unittest.TestCase):
    """Coverage for execution-stack generation and export behavior."""

    def trace_project_calling_external_module(self, package_name, only_project_root):
        with tempfile.TemporaryDirectory() as project_dir, tempfile.TemporaryDirectory() as external_dir:
            package_dir = os.path.join(external_dir, package_name)
            os.makedirs(package_dir, exist_ok=True)
            helper_path = os.path.join(package_dir, "helper.py")
            script_path = os.path.join(project_dir, "sample_script.py")

            with open(os.path.join(package_dir, "__init__.py"), "w", encoding="utf-8") as handle:
                handle.write("")
            with open(helper_path, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        """
                        def outside_func(value):
                            hidden = value + 10
                            return hidden
                        """
                    ).lstrip()
                )
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        f"""
                        from {package_name}.helper import outside_func

                        seed = 5
                        result = outside_func(seed)
                        """
                    ).lstrip()
                )

            with open(script_path, "r", encoding="utf-8") as handle:
                compiled = compile(handle.read(), script_path, "exec")

            vt = VarTracer(
                clean_stdlib=True,
                only_project_root=project_dir if only_project_root else None,
                verbose=False,
            )
            namespace = {"__name__": "__main__", "__file__": script_path}
            old_sys_path = list(sys.path)
            sys.path.insert(0, external_dir)
            for module_name in list(sys.modules):
                if module_name == package_name or module_name.startswith(package_name + "."):
                    del sys.modules[module_name]
            try:
                vt.start()
                exec(compiled, namespace)
            finally:
                vt.stop()
                sys.path[:] = old_sys_path
                for module_name in list(sys.modules):
                    if module_name == package_name or module_name.startswith(package_name + "."):
                        del sys.modules[module_name]

            exec_stack = vt.exec_stack_json(show_progress=False)
            dep_tree = vt.dep_tree_json(show_progress=False)
            return {
                "exec_stack": exec_stack,
                "dep_tree": dep_tree,
                "helper_path": helper_path,
                "script_path": script_path,
            }

    def test_exec_stack_contains_script_module_line_and_return_events(self):
        """A simple module should trace as CALL -> LINE(s) -> RETURN in the execution stack."""
        result = self.trace_source(
            """
            a = 1
            b = a + 2
            """
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        daughter_stack = root_call["details"]["daughter_stack"]

        self.assertEqual(root_call["details"]["depth"], 0)
        self.assertEqual(daughter_stack[0]["type"], "LINE")
        self.assertEqual(daughter_stack[0]["details"]["line_no"], "1")
        self.assertEqual(daughter_stack[0]["details"]["line_content"], "a = 1")
        self.assertEqual(daughter_stack[1]["type"], "LINE")
        self.assertEqual(daughter_stack[1]["details"]["line_no"], "2")
        self.assertEqual(daughter_stack[1]["details"]["line_content"], "b = a + 2")
        self.assertEqual(daughter_stack[-1]["type"], "RETURN")
        self.assertEqual(daughter_stack[-1]["details"]["func"], "<module>")

    def test_exec_stack_tracks_nested_function_call_depths(self):
        """Nested function calls should appear as daughter_stack entries with deeper depth."""
        result = self.trace_source(
            """
            def add(x, y):
                total = x + y
                return total

            result = add(1, 2)
            """
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        add_call = next(
            event
            for event in root_call["details"]["daughter_stack"]
            if event["type"] == "CALL" and event["details"]["func"] == "add"
        )

        self.assertEqual(add_call["details"]["depth"], 1)
        add_stack = add_call["details"]["daughter_stack"]
        self.assertEqual(add_stack[0]["type"], "LINE")
        self.assertEqual(add_stack[0]["details"]["depth"], 2)
        self.assertEqual(add_stack[0]["details"]["line_content"], "total = x + y")
        self.assertEqual(add_stack[1]["type"], "LINE")
        self.assertEqual(add_stack[1]["details"]["line_content"], "return total")
        self.assertEqual(add_stack[-1]["type"], "RETURN")
        self.assertEqual(add_stack[-1]["details"]["depth"], 2)

    def test_code_firstlineno_only_appears_on_call_events(self):
        """CALL events should identify their code object without duplicating the field elsewhere."""
        result = self.trace_source(
            """
            class Left:
                def __mul__(self, other):
                    return other

            class Right:
                def __mul__(self, other):
                    return other

            left = Left() * 1
            right = Right() * 2
            """
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        events = self.flatten_events([root_call])
        mul_calls = [
            event
            for event in events
            if event["type"] == "CALL" and event["details"]["func"] == "__mul__"
        ]

        self.assertEqual(len(mul_calls), 2)
        self.assertEqual(
            {event["details"]["code_firstlineno"] for event in mul_calls},
            {2, 6},
        )
        self.assertTrue(
            all("code_firstlineno" in event["details"] for event in events if event["type"] == "CALL")
        )
        self.assertTrue(
            all("code_firstlineno" not in event["details"] for event in events if event["type"] != "CALL")
        )

    def test_exec_stack_records_exception_event_details(self):
        """Raised exceptions should be serialized into EXCEPTION events with type and value."""
        result = self.trace_source(
            """
            try:
                1 / 0
            except ZeroDivisionError:
                handled = True
            """
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        exception_event = next(
            event for event in root_call["details"]["daughter_stack"] if event["type"] == "EXCEPTION"
        )

        self.assertEqual(exception_event["details"]["line_no"], "2")
        self.assertEqual(exception_event["details"]["line_content"], "1 / 0")
        self.assertEqual(exception_event["details"]["exception_type"], "ZeroDivisionError")
        self.assertIn("division by zero", exception_event["details"]["exception_value"])

    def test_clean_stdlib_filters_nested_stdlib_frames(self):
        """When clean_stdlib=True, stdlib implementation frames should stay out of the trace."""
        result = self.trace_source(
            """
            import pathlib
            value = pathlib.Path("a/b").as_posix()
            """
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        nested_events = self.flatten_events(root_call["details"]["daughter_stack"])
        stdlib_events = [
            event
            for event in nested_events
            if event.get("details", {}).get("module") == "pathlib"
        ]

        self.assertEqual(stdlib_events, [])

    def test_default_tracing_still_expands_project_external_python_modules(self):
        """Without only_project_root, non-stdlib modules outside the script directory keep current behavior."""
        result = self.trace_project_calling_external_module(
            "default_external_pkg",
            only_project_root=False,
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        nested_events = self.flatten_events(root_call["details"]["daughter_stack"])

        self.assertFalse(any(event.get("type") == "EXTERNAL_CALL" for event in nested_events))
        self.assertTrue(
            any(
                event.get("details", {}).get("file_path") == result["helper_path"]
                and event.get("details", {}).get("line_content") == "hidden = value + 10"
                for event in nested_events
            )
        )

    def test_only_project_root_collapses_external_python_modules_to_placeholder(self):
        """Project-root tracing should keep boundary calls without expanding external code internals."""
        result = self.trace_project_calling_external_module(
            "placeholder_external_pkg",
            only_project_root=True,
        )

        root_call = self.find_script_root_call(result["exec_stack"], result["script_path"])
        nested_events = self.flatten_events(root_call["details"]["daughter_stack"])
        external_events = [
            event for event in nested_events
            if event.get("type") == "EXTERNAL_CALL"
        ]

        self.assertTrue(external_events)
        self.assertIn(
            "external:placeholder_external_pkg.helper.outside_func",
            {event.get("details", {}).get("external_symbol") for event in external_events},
        )
        self.assertFalse(
            any(
                event.get("details", {}).get("file_path") == result["helper_path"]
                or event.get("details", {}).get("line_content") == "hidden = value + 10"
                for event in nested_events
            )
        )

        dep_tree = result["dep_tree"]
        external_symbol_ids = {
            symbol_id
            for symbol_id, symbol in dep_tree["symbols"].items()
            if symbol[0] == "external"
            and symbol[1] == "external:placeholder_external_pkg.helper.outside_func"
        }
        result_symbol_ids = {
            symbol_id
            for symbol_id, symbol in dep_tree["symbols"].items()
            if symbol[1] == "result"
        }

        self.assertTrue(external_symbol_ids)
        self.assertNotIn(result["helper_path"], set(dep_tree["files"].values()))
        external_files = {
            file_path
            for file_path in dep_tree["files"].values()
            if str(file_path).startswith("external:")
        }
        self.assertEqual(external_files, {"external:placeholder_external_pkg"})
        self.assertTrue(
            any(
                edge[0] in external_symbol_ids
                and edge[1] in result_symbol_ids
                and edge[2] == "ret"
                for edge in dep_tree["edges"]
            )
        )

    def test_raw_result_contains_script_call_line_and_return_markers(self):
        """The raw trace export should contain the expected textual event markers."""
        result = self.trace_source(
            """
            def add(x, y):
                return x + y

            result = add(1, 2)
            """
        )

        raw_result = result["raw_result"]

        self.assertIn(f"CALL - {result['script_path']}", raw_result)
        self.assertIn(f"LINE - {result['script_path']}:2 - add", raw_result)
        self.assertIn(f"RETURN - {result['script_path']}", raw_result)

    def test_exec_stack_json_writes_output_file(self):
        """The JSON execution-stack exporter should write the same structure it returns."""
        result = self.trace_source("value = 1\n")

        with tempfile.TemporaryDirectory() as output_dir:
            output = result["vt"].exec_stack_json(output_path=output_dir)
            output_file = os.path.join(output_dir, "VTrace_exec_stack.json")

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r", encoding="utf-8") as handle:
                written_text = handle.read()
                written = json.loads(written_text)

        self.assertEqual(output, written)
        self.assertEqual(written_text, json.dumps(output, separators=(",", ":")))
        self.assertIn("execution_stack", written)

    def test_exec_stack_txt_writes_human_readable_output(self):
        """The text execution-stack exporter should write a readable stack dump to disk."""
        result = self.trace_source(
            """
            def add(x, y):
                return x + y

            result = add(1, 2)
            """
        )

        with tempfile.TemporaryDirectory() as output_dir:
            output = result["vt"].exec_stack_txt(output_path=output_dir)
            output_file = os.path.join(output_dir, "VTrace_exec_stack.txt")

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r", encoding="utf-8") as handle:
                written = handle.read()

        self.assertEqual(output, written)
        self.assertIn("|CALL -", written)
        self.assertIn("|       func: add", written)
        self.assertIn("|       content: result = add(1, 2)", written)

    def test_dep_tree_json_writes_graph_output_file(self):
        """The dependency-tree exporter should persist the graph structure it returns."""
        result = self.trace_source(
            """
            a = 1
            b = a + 2
            """
        )

        with tempfile.TemporaryDirectory() as output_dir:
            output = result["vt"].dep_tree_json(output_path=output_dir)
            output_file = os.path.join(output_dir, "VTrace_dep_tree.json")

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r", encoding="utf-8") as handle:
                written = json.load(handle)

        self.assertEqual(output, written)
        self.assertIn("symbols", written)
        self.assertIn("edges", written)

    def test_dep_tree_edgelist_writes_compact_dependency_hints(self):
        """The dependency-tree text exporter should write the compact LLM hint format."""
        result = self.trace_source(
            """
            a = 1
            b = a + 2
            """
        )

        with tempfile.TemporaryDirectory() as output_dir:
            output = result["vt"].dep_tree_edgelist(output_path=output_dir)
            output_file = os.path.join(output_dir, "VTrace_dep_tree.edgelist")

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r", encoding="utf-8") as handle:
                written = handle.read()

        self.assertEqual(output, written)
        self.assertIn("# Unique Dependency Hints", written)
        self.assertIn("Use these dependency hints as supplemental evidence, not as final answers.", written)
        self.assertIn("- Retained files: `1`", written)
        self.assertIn("- Retained symbols: `2`", written)
        self.assertIn("- Retained edges: `1`", written)
        self.assertIn("- Matched dependency paths: `1`", written)
        self.assertIn("## Related File Clusters", written)
        self.assertIn("`sample_script.py`: appears in `1` retained dependency path(s)", written)
        self.assertIn("## Key Dependency Paths", written)
        self.assertIn("- path: `sample_script.py`", written)

    def test_flow_trace_contains_comment_frames_and_flow_steps(self):
        """The flow trace should expose compact runtime flow with documented step semantics."""
        result = self.trace_source(
            """
            def add(x, y):
                total = x + y
                return total

            result = add(1, 2)
            """
        )

        flow_trace = result["flow_trace"]
        self.assertIn("_comment", flow_trace)
        self.assertEqual(flow_trace["version"], "flow.v1")
        self.assertTrue(flow_trace["frames"])
        self.assertTrue(flow_trace["steps"])

        ops = [step[2] for step in flow_trace["steps"]]
        self.assertIn("call", ops)
        self.assertIn("ret", ops)
        self.assertIn("bind", ops)

    def test_flow_trace_records_conditional_reads_and_fallback_bind_steps(self):
        """Conditional tests and untraced calls should still surface as semantic flow steps."""
        result = self.trace_source(
            """
            x = 1
            if x > 0:
                y = x

            sep = "-"
            parts = ["a", "b"]
            text = sep.join(parts)
            """
        )

        flow_trace = result["flow_trace"]
        cond_steps = [step for step in flow_trace["steps"] if step[2] == "cond"]
        bind_steps = [step for step in flow_trace["steps"] if step[2] == "bind"]

        self.assertEqual(len(cond_steps), 1)
        self.assertEqual(cond_steps[0][3], 2)
        self.assertEqual(cond_steps[0][6]["stmt"], "If")

        fallback_bind = next(step for step in bind_steps if step[6].get("callee") == "sep.join")
        self.assertTrue(fallback_bind[6]["fallback"])
        self.assertEqual(fallback_bind[6]["stmt"], "Assign")

    def test_flow_trace_json_writes_output_file(self):
        """The flow-trace exporter should persist the same structure it returns."""
        result = self.trace_source(
            """
            def add(x, y):
                return x + y

            result = add(1, 2)
            """
        )

        with tempfile.TemporaryDirectory() as output_dir:
            output = result["vt"].flow_trace_json(output_path=output_dir)
            output_file = os.path.join(output_dir, "VTrace_flow_trace.json")

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r", encoding="utf-8") as handle:
                written = json.load(handle)

        self.assertEqual(output, written)
        self.assertIn("steps", written)
        self.assertIn("_comment", written)


if __name__ == "__main__":
    unittest.main()
