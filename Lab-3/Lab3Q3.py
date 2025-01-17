import torch
from matplotlib import pyplot as plt
a = torch.tensor(0.001)
class LinearRegression:
    def __init__(self):
        self.w = torch.tensor(1.0, requires_grad= True)
        self.b = torch.tensor(1.0, requires_grad= True)
    def forward(self, x):
        return (self.w * x) + self.b
    def loss(self, y_pred, y):
        return (y - y_pred) ** 2
    def update(self):
        self.w -= a * self.w.grad
        self.b -= a * self.b.grad
        return
    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
n = len(x)
model = LinearRegression()
losses = []
epochs = 100
for i in range(epochs):
    loss = 0.0
    for i in range(n):
        y_pred = model.forward(x[i])
        loss += model.loss(y_pred, y[i])
    loss /= n
    loss.backward()
    with torch.no_grad():
        model.update()
    print(f'Updated w = {model.w.item()},Updated b = {model.b.item()},Current Loss = {loss.item()}')
    print('-'*40)
    model.reset_grad()
    losses.append(loss.detach().numpy())
plt.plot(losses, label='Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs in Linear Regression Training')
plt.grid(True)
plt.legend()
plt.show()