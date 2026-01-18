import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import trace 

# 自动判断是否有 GPU 或 mps 可用
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义卷积神经网络
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

nnet = Nnet().to(device)  # 将模型移动到设备上

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nnet.parameters(), lr=0.001)

# 训练
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 移动数据到设备上

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

print("训练完成")




































# 测试
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"测试集准确率: {100 * correct / total:.2f}%")








# tracer = trace.Trace(count= 1, 
#                      trace=True, 
#                      )
# inputs = torch.randn(1, 3, 32, 32).to(device)  # 随机输入数据

# print("Tracing the model...")
# tracer.runfunc(net, inputs)
# # print("Number of files traced:", len(tracer.results.files()))
# tracer.results().write_results(
#     show_missing=1,
#     summary=False,
#     coverdir=r"C:\Users\jiuji\iCloudDrive\Documents\PhD\Working Documents\Deliverable 2\dataflow analysis and tool development\tool evaluation experiment\Empirical_Study_Tasks"
# )

# print("Tracing completed. Tracing report saved to trace_report.txt")


# import os
# print("当前工作目录为：", os.getcwd())