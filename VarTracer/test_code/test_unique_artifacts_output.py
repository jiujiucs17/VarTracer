import json
import os
import tempfile
import unittest

from VarTracer.Utilities import extract_unique_functions, filter_dep_tree_by_unique_artifacts


class TestUniqueArtifactsOutput(unittest.TestCase):
    def test_llm_text_uses_shared_symbols_for_unique_functions(self):
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
            with open(os.path.join(output_dir, "unique_artifacts.txt"), "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]

        self.assertEqual(sorted(payload.keys()), ["comparison", "unique_functions"])
        self.assertTrue(any(line.startswith("SYM ") for line in lines))

        fun_lines = [line for line in lines if line.startswith("FUN ")]
        self.assertEqual(len(fun_lines), 1)

        fun_record = json.loads(fun_lines[0][4:])
        self.assertEqual(fun_record["n"], "create_event")
        self.assertIn("mid", fun_record)
        self.assertIn("fid", fun_record)
        self.assertNotIn("mod", fun_record)
        self.assertNotIn("p", fun_record)

        sym_records = [json.loads(line[4:]) for line in lines if line.startswith("SYM ")]
        self.assertTrue(
            any(record.get("id") == fun_record["mid"] and record.get("k") == "mod" for record in sym_records)
        )
        self.assertTrue(
            any(record.get("id") == fun_record["fid"] and record.get("k") == "file" for record in sym_records)
        )

    def test_llm_text_can_filter_to_target_package(self):
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
            with open(os.path.join(output_dir, "unique_artifacts.txt"), "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]

        self.assertEqual(len(payload["unique_functions"]), 2)
        fun_lines = [json.loads(line[4:]) for line in lines if line.startswith("FUN ")]
        self.assertEqual(len(fun_lines), 1)
        self.assertEqual(fun_lines[0]["n"], "create_event")

        sym_lines = [json.loads(line[4:]) for line in lines if line.startswith("SYM ")]
        module_values = {line.get("v") for line in sym_lines if line.get("k") == "mod"}
        self.assertEqual(module_values, {"pandas.core.frame"})

    def test_dep_tree_edgelist_can_collapse_outside_target_package(self):
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
            with open(os.path.join(output_dir, "unique_dep_tree_edgelist.txt"), "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]

        self.assertIn("f2", filtered["files"])

        fil_lines = [json.loads(line[4:]) for line in lines if line.startswith("FIL ")]
        self.assertTrue(any(line.get("id") == "f_out" for line in fil_lines))
        self.assertFalse(any(line.get("p") == "/tmp/site-packages/numpy/core.py" for line in fil_lines))

        sym_lines = [json.loads(line[4:]) for line in lines if line.startswith("SYM ")]
        self.assertTrue(any(line.get("id") == "s_out" and line.get("f") == "f_out" for line in sym_lines))

        edg_lines = [json.loads(line[4:]) for line in lines if line.startswith("EDG ")]
        self.assertTrue(any(line.get("id") == "e_out" and line.get("src") == "s_out" and line.get("dst") == "s_out" for line in edg_lines))

        pth_lines = [json.loads(line[4:]) for line in lines if line.startswith("PTH ")]
        self.assertTrue(
            any(
                len(path.get("seq", [])) == 2
                and path["seq"][0] == "e_out"
                and path["seq"][1] != "e_out"
                for path in pth_lines
            )
        )

    def test_dep_tree_edgelist_deduplicates_identical_pth_records(self):
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
            with open(os.path.join(output_dir, "unique_dep_tree_edgelist.txt"), "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]

        pth_lines = [json.loads(line[4:]) for line in lines if line.startswith("PTH ")]
        self.assertEqual(
            sum(1 for path in pth_lines if path.get("sink") == "s2" and len(path.get("seq", [])) == 1),
            1,
        )


if __name__ == "__main__":
    unittest.main()
