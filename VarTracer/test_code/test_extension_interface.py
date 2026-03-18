import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from VarTracer import Utilities


class TestExtensionInterface(unittest.TestCase):
    """Tests for the subprocess-based extension entrypoint."""

    def find_var_info(self, file_deps, variable_name):
        matches = []
        for symbol_id, symbol_meta in file_deps.get("symbols", {}).items():
            if len(symbol_meta) < 5:
                continue
            kind, name, scope, file_id, line_no = symbol_meta
            if name == variable_name:
                matches.append(
                    {
                        "id": symbol_id,
                        "kind": kind,
                        "name": name,
                        "scope": scope,
                        "file_id": file_id,
                        "line": line_no,
                    }
                )
        self.assertTrue(matches, f"Expected to find variable '{variable_name}' in dependency output")
        return matches[0]

    def test_extension_interface_returns_dependency_payload(self):
        """The extension helper should return both exec_stack and dependency payloads."""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "extension_target.py")
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        """
                        a = 1
                        b = a + 3
                        """
                    ).lstrip()
                )

            original_run = subprocess.run

            def run_with_current_python(cmd, *args, **kwargs):
                # Force the subprocess to use the same interpreter and import path
                # as the current test process, so the test behaves like the real tool.
                if isinstance(cmd, list) and cmd and cmd[0] == "python3":
                    cmd = [sys.executable, *cmd[1:]]
                env = dict(os.environ)
                if env.get("PYTHONPATH"):
                    env["PYTHONPATH"] = os.pathsep.join([project_root, env["PYTHONPATH"]])
                else:
                    env["PYTHONPATH"] = project_root
                kwargs["env"] = env
                return original_run(cmd, *args, **kwargs)

            with mock.patch("VarTracer.Utilities.subprocess.run", side_effect=run_with_current_python):
                result = Utilities.extension_interface(script_path)

            self.assertIn("exec_stack", result)
            self.assertIn("dependency", result)
            self.assertIn(script_path, result["dependency"].get("files", {}).values())

            file_deps = result["dependency"]
            b_info = self.find_var_info(file_deps, "b")
            incoming_source_names = {
                file_deps["symbols"][edge[0]][1]
                for edge in file_deps.get("edges", [])
                if len(edge) >= 5 and edge[1] == b_info["id"]
            }
            self.assertIn("a", incoming_source_names)


if __name__ == "__main__":
    unittest.main()
