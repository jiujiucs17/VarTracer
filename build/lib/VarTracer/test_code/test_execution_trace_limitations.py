import unittest

from VarTracer.test_code._trace_test_support import TraceTestMixin


class TestExecutionTraceLimitations(TraceTestMixin, unittest.TestCase):
    """Regression coverage for trace-cleanliness issues that were previously broken."""

    def test_exec_stack_should_not_include_vartracer_internal_frames(self):
        """The public execution stack should not leak internal VarTracer implementation frames."""
        result = self.trace_source("value = 1\n")
        all_events = self.flatten_events(result["exec_stack"]["execution_stack"])

        tracer_events = [
            event
            for event in all_events
            if "VarTracer_Core.py" in event.get("details", {}).get("file_path", "")
        ]

        self.assertEqual(tracer_events, [])


if __name__ == "__main__":
    unittest.main()
