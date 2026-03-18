import unittest

from VarTracer.test_code._trace_test_support import TraceTestMixin


class TestVarTracerFlow(TraceTestMixin, unittest.TestCase):
    """Integration tests for the end-to-end data dependency pipeline."""

    def test_dep_tree_captures_simple_assignment(self):
        """A minimal script should produce a dependency tree where b depends on a."""
        result = self.trace_source(
            """
            a = 1
            b = a + 2
            """
        )

        dep_tree = result["dep_tree"]
        a_info = self.find_var_info(dep_tree, "a")
        b_info = self.find_var_info(dep_tree, "b")

        self.assertEqual(self.incoming_edges(dep_tree, a_info["id"]), [])
        b_incoming = self.incoming_edges(dep_tree, b_info["id"])
        self.assertEqual(len(b_incoming), 1)
        self.assertIn("a", self.incoming_source_names(dep_tree, b_info["id"]))
        self.assertEqual(b_incoming[0][3], 2)
        self.assertEqual(b_incoming[0][4], 1)
        self.assertIn([a_info["id"], b_info["id"]], dep_tree["paths"].get(b_info["id"], []))

    def test_dep_tree_captures_subscript_assignment(self):
        """Subscript writes should be represented as container-element assignments."""
        result = self.trace_source(
            """
            items = [1, 2, 3]
            items[0] = 99
            """
        )

        dep_tree = result["dep_tree"]
        target_info = self.find_var_info(dep_tree, "items[]")

        self.assertIn("items", self.incoming_source_names(dep_tree, target_info["id"]))


if __name__ == "__main__":
    unittest.main()
