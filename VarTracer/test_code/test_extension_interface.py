from VarTracer import Utilities
import json

file_path = r"/Users/zhangmengqi/Desktop/test_case_for_extension/playground.py"

output = Utilities.extension_interface(file_path)
print("Output from extension_interface:")
print(json.dumps(output, indent=4))