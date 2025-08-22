from VarTracer_Core import *
import json

# output_folder = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_2'
# print("Comparing execution stacks...")
# exec_stack_json_1_path = output_folder + "/exec_stack" + "/exec_stack_task_7_1.json"
# exec_stack_json_0_path = output_folder + "/exec_stack" + "/exec_stack_task_7_0.json"
# dep_tree_json_1_path = output_folder + "/dependency" + "/dep_tree_task_7_1.json"
# dep_tree_json_0_path = output_folder + "/dependency" + "/dep_tree_task_7_0.json"

output_folder = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_3'
print("Comparing execution stacks...")
exec_stack_json_1_path = output_folder + "/exec_stack" + "/exec_stack_task_8_1.json"
exec_stack_json_0_path = output_folder + "/exec_stack" + "/exec_stack_task_8_0.json"
dep_tree_json_1_path = output_folder + "/dependency" + "/dep_tree_task_8_1.json"
dep_tree_json_0_path = output_folder + "/dependency" + "/dep_tree_task_8_0.json"

# output_folder = '/Users/zhangmengqi/Documents/PhD/Working Documents/Deliverable 2/dataflow analysis and tool development/tool evaluation experiment/Pilot_Study_3'
# print("Comparing execution stacks...")
# exec_stack_json_1_path = output_folder + "/exec_stack" + "/exec_stack_task_9_1.json"
# exec_stack_json_0_path = output_folder + "/exec_stack" + "/exec_stack_task_9_0.json"
# dep_tree_json_1_path = output_folder + "/dependency" + "/dep_tree_task_9_1.json"
# dep_tree_json_0_path = output_folder + "/dependency" + "/dep_tree_task_9_0.json"

# read json file from json document
with open(exec_stack_json_1_path, 'r', encoding='utf-8') as f:
    exec_stack_json_1 = json.load(f)

with open(exec_stack_json_0_path, 'r', encoding='utf-8') as f:
    exec_stack_json_0 = json.load(f)

# read json file from json document
with open(dep_tree_json_1_path, 'r', encoding='utf-8') as f:
    dep_tree_json_1 = json.load(f)

with open(dep_tree_json_0_path, 'r', encoding='utf-8') as f:
    dep_tree_json_0 = json.load(f)

compare_exec_stack(exec_stack_json_0, 
                   exec_stack_json_1, 
                   dep_tree_json_0, 
                   dep_tree_json_1, 
                   output_folder)
print("Execution stacks compared and saved to compare_exec_stack_result.xlsx")




