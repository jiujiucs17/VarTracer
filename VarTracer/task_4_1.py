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
        # 将 nn.Linear 替换为 nn.Conv1d
        self.conv1 = nn.Conv1d(8, 4, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, in_channels, width)
        x = self.conv1(x)
        return x

model = SimpleModel()
# 输入 shape: (batch_size, in_channels, width)
x = torch.randn(1, 8, 5)
with torch.no_grad():
    y = model(x)
print("Output with conv1d:", y)

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

print("Generating dependency tree JSON...")
dependency_output_path = f"{output_path}/dependency"
vt.dep_tree_json(output_path=dependency_output_path, output_name="dep_tree_task_4_1.json")

print("Execution stack JSON generated")
# -------------- VarTracer related code --------------
