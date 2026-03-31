import json
import os
import tempfile
import unittest

from VarTracer.Utilities import extract_unique_functions


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


if __name__ == "__main__":
    unittest.main()
