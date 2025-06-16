from VarTracer_Core import *
import json

output_folder = "/Users/zhangmengqi/Desktop/test_case_for_extension"

# compare execution stacks
print("Comparing execution stacks...")
exec_stack_json_1_path = output_folder + "/exec_stack" + "/exec_stack_task_1_1.json"
exec_stack_json_0_path = output_folder + "/exec_stack" + "/exec_stack_task_1_0.json"

# read json file from json document
with open(exec_stack_json_1_path, 'r', encoding='utf-8') as f:
    exec_stack_json_1 = json.load(f)

with open(exec_stack_json_0_path, 'r', encoding='utf-8') as f:
    exec_stack_json_0 = json.load(f)

compare_exec_stack(exec_stack_json_1, exec_stack_json_0, output_folder, "compare_exec_stack_result.xlsx")
print("Execution stacks compared and saved to compare_exec_stack_result.xlsx")

# comapre dependencies
print("Comparing dependencies...")
dependency_tree_1_path = output_folder + "/dependency" + "/dep_tree_task_1_1.json"
dependency_tree_0_path = output_folder + "/dependency" + "/dep_tree_task_1_0.json"

# read xlsx file from xlsx document
with open(dependency_tree_1_path, 'r', encoding='utf-8') as f:
    dependency_tree_1 = json.load(f)

with open(dependency_tree_0_path, 'r', encoding='utf-8') as f:
    dependency_tree_0 = json.load(f)

compare_dependency(dependency_tree_1, dependency_tree_0, output_folder, "compare_dependency_tree_result.xlsx")
print("Dependencies compared and saved to compare_dependency_tree_result.xlsx")
