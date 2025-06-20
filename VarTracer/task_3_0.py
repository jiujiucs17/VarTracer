import torch
import torch.nn as nn

# -------------- VarTracer related code --------------
from VarTracer_Core import *
import json
vt = VarTracer()
vt.start()
# -------------- VarTracer related code --------------

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.linear1(x)
        return x

model = SimpleModel()
x = torch.randn(1, 8)
with torch.no_grad():
    y = model(x)
print("Output without extra linear:", y)

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
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_3_0.json", show_progress=True)

print("Execution stack JSON generated")
# -------------- VarTracer related code --------------
