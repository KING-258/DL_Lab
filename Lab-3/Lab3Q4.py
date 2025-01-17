import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from matplotlib import pyplot as plt
X = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
class LinearRegressionDataset(data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, ind):
        return self.X[ind], self.y[ind]
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
dataset = LinearRegressionDataset(X, y)
batch_size = dataset.__len__()
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle= True)
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr= 0.001)
loss_list = []
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for input, target in dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'parameters = w = {model.linear.weight.item()}, b = {model.linear.bias.item()}, loss = {loss.item()}')
    loss_list.append(running_loss)
plt.plot(loss_list)
plt.show()