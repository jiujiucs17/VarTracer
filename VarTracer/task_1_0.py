# This is task 1_0, without the torchscript code
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import trace 

from VarTracer_Core import *
import json
vt = VarTracer()
vt.start()

# Automatically detect if GPU or MPS is available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing and loading
print("Starting to load dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR-10 training set...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

print("Loading CIFAR-10 test set...")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("Dataset loading completed")

print("Defining convolutional neural network...")
# Define convolutional neural network
class Nnet(nn.Module):
    def __init__(self):
        super(Nnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
print("Convolutional neural network defined")
print("Moving model to device...")
nnet = Nnet().to(device)  # Move model to device
print("Model moved to device")

print("Defining loss function and optimizer...")
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nnet.parameters(), lr=0.001)
print("Loss function and optimizer defined")

# Training
print("Starting model training...")
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if i > 0:
            break
        inputs, labels = data[0].to(device), data[1].to(device)  # Move data to device
        

        optimizer.zero_grad()
        outputs = nnet(inputs)

        # Alternatives for line 57 without pytorch:
            # outputs = nnet.forwar(inputs)
            # nnet.connect_pre_hooks
            # nnet.connect_hooks
            # nnet.compile(forward)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Training completed")
vt.stop()
# print("Length of raw_log", len(vt.raw_logs))
# print("Generating execution stack and dependency tree...")
# exec_stack_json = vt.exec_stack_json(show_progress=True)
# print("Execution stack generated")
# dep_tree = DependencyTree(call_stack=json.dumps(exec_stack_json))
# dep_dic = dep_tree.parse_dependency()
# print("Dependency tree generated")

# result_json = {
#     'exec_stack': exec_stack_json,
#     'dependency': dep_dic
# }
# print("Result JSON generated")
# # with open(r'/Users/zhangmengqi/Desktop/test_case_for_extension/dep_tree.json', 'w') as result_file:
# #     json.dump(dep_dic, result_file)
# #     print("Result saved to dep_tree.json")

output_path = '/Users/zhangmengqi/Desktop/test_case_for_extension'
print("Generating execution stack and dependency tree...")
# vt.exec_stack_txt(output_path)
# print("Execution stack generated")
# vt.dep_tree_xlsx(output_path)
# print("Dependency tree generated")

print("Generating execution stack JSON...")
exec_stack_json_output_path = f"{output_path}/exec_stack"
vt.exec_stack_json(output_path=exec_stack_json_output_path, output_name="exec_stack_task_1_0.json", show_progress=True)

print("Execution stack JSON generated")
print("Generating dependency tree JSON...")
dependency_output_path = f"{output_path}/dependency"
vt.dep_tree_json(output_path=dependency_output_path, output_name="dep_tree_task_1_0.json")
print("Dependency tree JSON generated")