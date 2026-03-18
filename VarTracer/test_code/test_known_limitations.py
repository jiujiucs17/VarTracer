import unittest

from VarTracer.ASTParser import LineDependencyAnalyzer
from VarTracer.test_code._trace_test_support import TraceTestMixin


class TestKnownLimitations(TraceTestMixin, unittest.TestCase):
    """Regression tests for dependency cases that used to fail before the repair."""

    def test_multiline_assignment_should_keep_dependencies(self):
        """A multiline assignment should still resolve dependencies from the whole statement."""
        result = self.trace_source(
            """
            a = 1
            b = 2
            c = (
                a +
                b
            )
            """
        )

        dep_tree = result["dep_tree"]
        c_info = self.find_var_info(dep_tree, "c")
        c_sources = self.incoming_source_names(dep_tree, c_info["id"])
        self.assertIn("a", c_sources)
        self.assertIn("b", c_sources)

    def test_augassign_should_depend_on_previous_value(self):
        """Augmented assignment must treat the target as both read and written."""
        result = LineDependencyAnalyzer({"a": 1}, {}).analyze("a += 1")

        self.assertEqual(result["assigned_vars"], {"a"})
        self.assertIn("a", result["dependencies"])

    def test_walrus_assignment_should_create_named_binding(self):
        """A walrus expression in an if-header should create n and link d back to n."""
        result = self.trace_source(
            """
            y = [1]
            if (n := len(y)) > 0:
                d = n
            """
        )

        dep_tree = result["dep_tree"]
        n_info = self.find_var_info(dep_tree, "n")
        d_info = self.find_var_info(dep_tree, "d")

        self.assertIn("y", self.incoming_source_names(dep_tree, n_info["id"]))
        self.assertIn("n", self.incoming_source_names(dep_tree, d_info["id"]))

    def test_comprehension_should_not_duplicate_outer_assignment(self):
        """Comprehension loop variables should not be confused with outer-scope dependencies."""
        result = self.trace_source(
            """
            x = 100
            y = [x for x in range(3)]
            """
        )

        dep_tree = result["dep_tree"]
        y_entries = [self.find_var_info(dep_tree, "y")]

        self.assertEqual(len(y_entries), 1)
        y_sources = self.incoming_source_names(dep_tree, y_entries[0]["id"])
        self.assertNotIn("x", y_sources)
        self.assertIn("range", y_sources)

    def test_chain_attributes_assign_should_contain_all_func_names(self):
        result = self.trace_source(
            """
            import toyarray as np
            X = np.arange(12).reshape(6, 2)
            """,
            extra_files={
                "toyarray/__init__.py": """
                class ToyArray:
                    def __init__(self, values):
                        self.values = values

                    def reshape(self, rows, cols):
                        return [
                            self.values[row * cols:(row + 1) * cols]
                            for row in range(rows)
                        ]

                def arange(count):
                    return ToyArray(list(range(count)))
                """
            },
        )
        
        dep_tree = result["dep_tree"]
        X_info = self.find_var_info(dep_tree, "X")
        X_sources = self.incoming_source_names(dep_tree, X_info["id"])
        self.assertIn("np.arange", X_sources)
        self.assertIn("np.reshape", X_sources)
        self.assertIn("np", X_sources)


if __name__ == "__main__":
    unittest.main()
