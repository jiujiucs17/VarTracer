import torch

# -------------- VarTracer related code --------------
from VarTracer_Core import *
import json
vt = VarTracer()
vt.start()
# -------------- VarTracer related code --------------

x = torch.arange(20).reshape(4, 5)
# y = x[1:3, ::2]  # Slice rows 1-2 and every other column
z = x.view(-1)   # Flatten

# -------------- VarTracer related code --------------
vt.stop()
output_path = '/Users/zhangmengqi/Desktop/test_case_for_extension'
print("Generating execution stack and dependency tree...")
# vt.exec_stack_txt(output_path)
# print("Execution stack generated")
# vt.dep_tree_xlsx(output_path)
# print("Dependency tree generated")

print("Generating execution stack JSON...")
exec_stack_json_output_path = f"{output_path}/exec_stack"
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_4_1.json", show_progress=True)

print("Execution stack JSON generated")
# -------------- VarTracer related code --------------
