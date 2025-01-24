import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
class XORDataset(Dataset):
    def __init__(self, X1, X2, y):
        super().__init__()
        self.X1 = X1.reshape(-1, 1)
        self.X2 = X2.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
    def __len__(self):
        return self.X1.shape[0]
    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.y[index]
class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2, bias= True)
        self.activate1 = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1, bias= True)
        self.activate2 = nn.Sigmoid()
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim= 1)
        z1 = self.linear1(x)
        a1 = self.activate1(z1)
        z2 = self.linear2(a1)
        a2 = self.activate2(z2)
        return a2
def main():
    X1 = torch.tensor([0, 0, 1, 1], dtype= torch.float32)
    X2 = torch.tensor([0, 1, 0, 1], dtype= torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype= torch.float32)
    model = XORModel()
    dataset = XORDataset(X1=X1, X2=X2, y=y)
    dataloader = DataLoader(dataset=dataset, batch_size=dataset.__len__(), shuffle= True)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)
    loss_list = []
    model.train()
    epochs = 100
    for epoch in range(epochs):
        loss = 0.0
        for input1, input2, target in dataloader:
            optimizer.zero_grad()
            output = model(input1, input2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            a = loss
            loss_list.append(a.detach().numpy())
            if epoch % 25 == 0:
                print(f'Epoch : {epoch}\nCurrent Loss = {loss.item()}')
    plt.plot(loss_list)
    plt.show()
if __name__ == '__main__':
    main()