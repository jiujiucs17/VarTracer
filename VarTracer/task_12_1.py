import tensorflow as tf

# 创建一个数据点
x = tf.constant([[2.0, -1.0, 0.5]])

# -------------- VarTracer related code --------------
from VarTracer_Core import *
import json
vt = VarTracer()
vt.start()
# -------------- VarTracer related code --------------

# 计算均值（上下文）
mean_x = tf.reduce_mean(x)

# 用均值中心化后ReLU
y = tf.nn.relu(x - mean_x)

# -------------- VarTracer related code --------------
vt.stop()
output_path = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_6'

print("Generating execution stack and dependency tree...")
# vt.exec_stack_txt(output_path)
# print("Execution stack generated")
# vt.dep_tree_xlsx(output_path)
# print("Dependency tree generated")

print("Generating execution stack JSON...")
exec_stack_json_output_path = f"{output_path}/exec_stack"
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_12_1.json", show_progress=True)

print("Generating dependency tree JSON...")
dependency_output_path = f"{output_path}/dependency"
vt.dep_tree_json(output_path=dependency_output_path, output_name="dep_tree_task_12_1.json")

print("Execution stack JSON and dependency tree JSON generated")
# -------------- VarTracer related code --------------