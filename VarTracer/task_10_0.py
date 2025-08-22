import mindspore
from mindspore import nn
from mindspore import context
import mindspore.numpy as mnp

# -------------- VarTracer related code --------------
from VarTracer_Core import *
import json
vt = VarTracer()
vt.start()
# -------------- VarTracer related code --------------


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm((512,))

    def construct(self, x):
        x = self.relu(x)
        # x = self.ln(x)
        return x
    
model = Network()
x = mindspore.Tensor(mindspore.numpy.randn(1, 512), mindspore.float32)
y = model(x)
print(y.shape)

# -------------- VarTracer related code --------------
vt.stop()
# output_path = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_2'
output_path = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_4'

print("Generating execution stack and dependency tree...")
# vt.exec_stack_txt(output_path)
# print("Execution stack generated")
# vt.dep_tree_xlsx(output_path)
# print("Dependency tree generated")

print("Generating execution stack JSON...")
exec_stack_json_output_path = f"{output_path}/exec_stack"
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_10_0.json", show_progress=True)

print("Generating dependency tree JSON...")
dependency_output_path = f"{output_path}/dependency"
vt.dep_tree_json(output_path=dependency_output_path, output_name="dep_tree_task_10_0.json")

print("Execution stack JSON and dependency tree JSON generated")
# -------------- VarTracer related code --------------