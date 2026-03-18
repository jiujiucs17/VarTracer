import unittest

from VarTracer.ASTParser import LineDependencyAnalyzer


class TestLineDependencyAnalyzer(unittest.TestCase):
    """Unit tests for the smallest dependency-analysis building block."""

    def analyze(self, code_line, local_vars=None, global_vars=None):
        analyzer = LineDependencyAnalyzer(local_vars or {}, global_vars or {})
        return analyzer.analyze(code_line)

    def test_simple_assignment_tracks_dependency_and_assignment(self):
        """Verify that a basic assignment records both the target and the RHS dependency."""
        result = self.analyze("b = a + 5", local_vars={"a": 1})

        self.assertEqual(result["assigned_vars"], {"b"})
        self.assertEqual(result["dependencies"], {"a"})

    def test_tuple_unpacking_tracks_multiple_targets(self):
        """Verify that unpacking assigns to every target while preserving the source dependency."""
        result = self.analyze("left, right = source", local_vars={"source": (1, 2)})

        self.assertEqual(result["assigned_vars"], {"left", "right"})
        self.assertEqual(result["dependencies"], {"source"})

    def test_attribute_assignment_tracks_loaded_values(self):
        """Verify that attribute stores still keep dependencies from the loaded RHS container."""
        result = self.analyze(
            "obj.value = items[0]",
            local_vars={"obj": object(), "items": [1, 2, 3]},
        )

        self.assertEqual(result["assigned_vars"], {"obj.value"})
        self.assertEqual(result["dependencies"], {"items"})


if __name__ == "__main__":
    unittest.main()
