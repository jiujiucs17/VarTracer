import os
import tempfile
import unittest

from VarTracer.Utilities import extract_unique_functions, filter_dep_tree_by_unique_artifacts


class TestUniqueArtifactsOutput(unittest.TestCase):
    def _read_output(self, output_dir, file_name):
        with open(os.path.join(output_dir, file_name), "r", encoding="utf-8") as handle:
            return handle.read()

    def test_unique_artifacts_llm_text_uses_markdown_summary(self):
        utilities_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Utilities.py")
        )
        negative = {
            "execution_stack": [
                {
                    "type": "CALL",
                    "details": {
                        "module": "VarTracer.Utilities",
                        "file_path": utilities_path,
                        "func": "safe_serialize",
                        "line_no": 14,
                        "depth": 0,
                        "daughter_stack": [
                            {
                                "type": "LINE",
                                "details": {
                                    "module": "VarTracer.Utilities",
                                    "file_path": utilities_path,
                                    "func": "safe_serialize",
                                    "line_no": 17,
                                    "depth": 1,
                                    "line_content": "return str(obj)",
                                },
                            }
                        ],
                    },
                }
            ]
        }
        positive = {
            "execution_stack": [
                negative["execution_stack"][0],
                {
                    "type": "CALL",
                    "details": {
                        "module": "VarTracer.Utilities",
                        "file_path": utilities_path,
                        "func": "create_event",
                        "line_no": 23,
                        "depth": 0,
                        "daughter_stack": [
                            {
                                "type": "LINE",
                                "details": {
                                    "module": "VarTracer.Utilities",
                                    "file_path": utilities_path,
                                    "func": "create_event",
                                    "line_no": 25,
                                    "depth": 1,
                                    "line_content": "event = {'type': event_type, 'details': base_info}",
                                },
                            }
                        ],
                    },
                },
            ]
        }

        with tempfile.TemporaryDirectory() as output_dir:
            payload = extract_unique_functions(
                positive,
                negative,
                output_dir,
                generate_llm_txt=True,
            )
            text = self._read_output(output_dir, "unique_artifacts.txt")

        self.assertEqual(sorted(payload.keys()), ["comparison", "unique_functions"])
        self.assertIn("# Unique Artifacts", text)
        self.assertIn("## Related Files", text)
        self.assertIn("## Artifact Candidates", text)
        self.assertIn("artifact: `VarTracer/Utilities.py:create_event`", text)
        self.assertIn("trace activity:", text)
        self.assertIn("sample executed lines:", text)
        self.assertNotIn("SYM ", text)
        self.assertNotIn("FUN ", text)

    def test_unique_artifacts_llm_text_can_filter_to_target_package(self):
        utilities_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Utilities.py")
        )
        negative = {"execution_stack": []}
        positive = {
            "execution_stack": [
                {
                    "type": "CALL",
                    "details": {
                        "module": "pandas.core.frame",
                        "file_path": utilities_path,
                        "func": "create_event",
                        "line_no": 23,
                        "depth": 0,
                        "daughter_stack": [
                            {
                                "type": "LINE",
                                "details": {
                                    "module": "pandas.core.frame",
                                    "file_path": utilities_path,
                                    "func": "create_event",
                                    "line_no": 25,
                                    "depth": 1,
                                    "line_content": "event = {'type': event_type, 'details': base_info}",
                                },
                            }
                        ],
                    },
                },
                {
                    "type": "CALL",
                    "details": {
                        "module": "numpy.core.multiarray",
                        "file_path": utilities_path,
                        "func": "safe_serialize",
                        "line_no": 14,
                        "depth": 0,
                        "daughter_stack": [
                            {
                                "type": "LINE",
                                "details": {
                                    "module": "numpy.core.multiarray",
                                    "file_path": utilities_path,
                                    "func": "safe_serialize",
                                    "line_no": 17,
                                    "depth": 1,
                                    "line_content": "return str(obj)",
                                },
                            }
                        ],
                    },
                },
            ]
        }

        with tempfile.TemporaryDirectory() as output_dir:
            payload = extract_unique_functions(
                positive,
                negative,
                output_dir,
                generate_llm_txt=True,
                target_package="pandas",
            )
            text = self._read_output(output_dir, "unique_artifacts.txt")

        self.assertEqual(len(payload["unique_functions"]), 2)
        self.assertIn("Target package filter: `pandas`", text)
        self.assertIn("artifact: `pandas/core/frame.py:create_event`", text)
        self.assertNotIn("safe_serialize", text)
        self.assertNotIn("numpy.core.multiarray", text)

    def test_dep_tree_llm_text_uses_markdown_and_hides_external_details(self):
        unique_artifacts = {
            "comparison": {
                "trace_a_role": "feature_positive",
                "trace_b_role": "feature_negative",
                "comparison_type": "trace_a_minus_trace_b",
                "unique_function_count": 1,
            },
            "unique_functions": [
                {
                    "name": "target_func",
                    "qualified_name": "target_func",
                    "module_name": "pandas.core.frame",
                    "path": "/tmp/site-packages/pandas/core/frame.py",
                    "relative_path": "pandas/core/frame.py",
                    "file_name": "frame.py",
                    "defined_line_number": 10,
                    "parent_scope": None,
                    "class_name": None,
                }
            ],
        }
        dep_tree = {
            "version": "1",
            "trace_started_at": "now",
            "files": {
                "f1": "/tmp/site-packages/pandas/core/frame.py",
                "f2": "/tmp/site-packages/numpy/core.py",
            },
            "symbols": {
                "s1": ["func", "target_func", "<module>", "f1", 10],
                "s2": ["var", "value", "target_func", "f1", 11],
                "s3": ["var", "other1", "helper", "f2", 20],
                "s4": ["var", "other2", "helper", "f2", 21],
            },
            "edges": [
                ["s3", "s4", "use", 21, 1],
                ["s4", "s2", "arg", 21, 1],
                ["s1", "s2", "define", 11, 1],
            ],
            "paths": {
                "s2": [["s3", "s4", "s2"], ["s1", "s2"]],
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            filtered = filter_dep_tree_by_unique_artifacts(
                unique_artifacts,
                dep_tree,
                output_dir,
                generate_llm_txt=True,
                target_package="pandas",
            )
            text = self._read_output(output_dir, "unique_dep_tree_edgelist.txt")

        self.assertIn("f2", filtered["files"])
        self.assertIn("# Unique Dependency Hints", text)
        self.assertIn("## Related File Clusters", text)
        self.assertIn("## Key Dependency Paths", text)
        self.assertIn("## Anchor Artifacts", text)
        self.assertIn("anchor: `pandas/core/frame.py:target_func`", text)
        self.assertIn("[outside target package omitted]", text)
        self.assertNotIn("/tmp/site-packages/numpy/core.py", text)
        self.assertNotIn("helper", text)
        self.assertNotIn("FIL ", text)
        self.assertNotIn("SYM ", text)
        self.assertNotIn("EDG ", text)
        self.assertNotIn("PTH ", text)

    def test_dep_tree_llm_text_deduplicates_identical_paths(self):
        unique_artifacts = {
            "comparison": {
                "trace_a_role": "feature_positive",
                "trace_b_role": "feature_negative",
                "comparison_type": "trace_a_minus_trace_b",
                "unique_function_count": 1,
            },
            "unique_functions": [
                {
                    "name": "target_func",
                    "qualified_name": "target_func",
                    "module_name": "pandas.core.frame",
                    "path": "/tmp/site-packages/pandas/core/frame.py",
                    "relative_path": "pandas/core/frame.py",
                    "file_name": "frame.py",
                    "defined_line_number": 10,
                    "parent_scope": None,
                    "class_name": None,
                }
            ],
        }
        dep_tree = {
            "version": "1",
            "trace_started_at": "now",
            "files": {
                "f1": "/tmp/site-packages/pandas/core/frame.py",
            },
            "symbols": {
                "s1": ["func", "target_func", "<module>", "f1", 10],
                "s2": ["var", "value", "target_func", "f1", 11],
            },
            "edges": [
                ["s1", "s2", "define", 11, 1],
            ],
            "paths": {
                "s2": [["s1", "s2"], ["s1", "s2"]],
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            filter_dep_tree_by_unique_artifacts(
                unique_artifacts,
                dep_tree,
                output_dir,
                generate_llm_txt=True,
                target_package="pandas",
            )
            text = self._read_output(output_dir, "unique_dep_tree_edgelist.txt")

        self.assertEqual(text.count("- path: `pandas/core/frame.py:target_func`"), 1)


if __name__ == "__main__":
    unittest.main()
