import unittest

from VarTracer.test_code._trace_test_support import TraceTestMixin


class TestHistoricalBugs(TraceTestMixin, unittest.TestCase):
    """Regression tests for bugs fixed after being observed in real tracing runs."""

    def test_frame_snapshot_keeps_only_names_without_breaking_dependency_analysis(self):
        """
        Frame snapshots should store only scope names, while dependency extraction
        still resolves local and global reads correctly.
        """
        result = self.trace_source(
            """
            global_value = 10

            def compute(local_input):
                local_temp = local_input + 1
                output = local_temp + global_value
                return output

            final_value = compute(3)
            """
        )

        line_frames = [
            record["frame"]
            for record in result["vt"].raw_logs
            if record["event"] == "line" and record["frame"].function_name == "compute"
        ]
        self.assertTrue(line_frames)

        for frame in line_frames:
            self.assertIsInstance(frame.locals, list)
            self.assertIsInstance(frame.globals, list)
            self.assertTrue(all(isinstance(name, str) for name in frame.locals))
            self.assertTrue(all(isinstance(name, str) for name in frame.globals))
            self.assertFalse(hasattr(frame, "code_context"))

        compute_output = self.find_var_info(result["dep_tree"], "output")
        self.assertEqual(
            self.incoming_source_names(result["dep_tree"], compute_output["id"]),
            {"global_value", "local_temp"},
        )

    def test_trace_survives_import_of_module_with_half_initialized_global_str(self):
        """
        FrameSnapshot should not crash when a traced import exposes a global object
        whose __str__ depends on a later module-level binding.
        """
        result = self.trace_source(
            """
            import half_initialized_module
            value = half_initialized_module.marker
            """,
            extra_files={
                "half_initialized_module.py": """
                class LateString:
                    def __str__(self):
                        return marker

                obj = LateString()
                x = 1
                marker = "ready"
                """
            },
        )

        all_events = self.flatten_events(result["exec_stack"]["execution_stack"])
        imported_module_events = [
            event
            for event in all_events
            if event.get("details", {}).get("file_path", "").endswith("half_initialized_module.py")
        ]

        self.assertTrue(imported_module_events)
        self.assertTrue(
            any(
                event.get("type") == "RETURN"
                and event.get("details", {}).get("func") == "<module>"
                for event in imported_module_events
            )
        )
        self.assertIn("half_initialized_module.py", result["raw_result"])


if __name__ == "__main__":
    unittest.main()
