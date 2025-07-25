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
        x = x
        return x

model = SimpleModel()
x = torch.randn(1, 8)
with torch.no_grad():
    y = model(x)
print("Output without linear:", y)

# -------------- VarTracer related code --------------
vt.stop()
output_path = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_1'
print("Generating execution stack and dependency tree...")
# vt.exec_stack_txt(output_path)
# print("Execution stack generated")
# vt.dep_tree_xlsx(output_path)
# print("Dependency tree generated")

print("Generating execution stack JSON...")
exec_stack_json_output_path = f"{output_path}/exec_stack"
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_3_0.json", show_progress=True)

print("Generating dependency tree JSON...")
dependency_output_path = f"{output_path}/dependency"
vt.dep_tree_json(output_path=dependency_output_path, output_name="dep_tree_task_3_0.json")

print("Execution stack JSON and dependency tree JSON generated")
# -------------- VarTracer related code --------------
