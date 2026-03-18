import json
import os
import tempfile
import unittest

from VarTracer.test_code._trace_test_support import TraceTestMixin


class TestExecutionTrace(TraceTestMixin, unittest.TestCase):
    """Coverage for execution-stack generation and export behavior."""

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
                written = json.load(handle)

        self.assertEqual(output, written)
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

    def test_dep_tree_edgelist_writes_compact_text_output(self):
        """The edge-list exporter should render readable EDGE and PATH lines."""
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
        self.assertIn("EDGE <module>::a<var> -> <module>::b<var> [data@2]", written)
        self.assertIn("PATH <module>::b<var> <=", written)


if __name__ == "__main__":
    unittest.main()
