# this is task_2_0, not using the autograd functionality of pytorch

import torch
import torch.nn as nn

# -------------- VarTracer related code --------------
from VarTracer_Core import *
import json
vt = VarTracer()
vt.start()
# -------------- VarTracer related code --------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
x = torch.randn(16, 32)  # 没有 requires_grad
target = torch.randint(0, 10, (16,))
criterion = nn.CrossEntropyLoss()

with torch.no_grad():  # 确保无 autograd 构图
    output = model(x)
    loss = criterion(output, target)
print("Forward done, no grad involved")

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
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_2_0.json", show_progress=True)

print("Execution stack JSON generated")
# -------------- VarTracer related code --------------
