# this is task_2_1, using the autograd functionality of pytorch

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

# 初始化模型与数据
model = MLP()
x = torch.randn(16, 32, requires_grad=True)  # 启用 autograd
target = torch.randint(0, 10, (16,))
criterion = nn.CrossEntropyLoss()

# 前向与反向
output = model(x)
loss = criterion(output, target)
grads = torch.autograd.grad(loss, model.parameters())  # 使用 autograd 引擎
print("Computed gradients")

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
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_2_1.json", show_progress=True)

print("Generating dependency tree JSON...")
dependency_output_path = f"{output_path}/dependency"
vt.dep_tree_json(output_path=dependency_output_path, output_name="dep_tree_task_2_1.json")

print("Execution stack JSON generated")
# -------------- VarTracer related code --------------