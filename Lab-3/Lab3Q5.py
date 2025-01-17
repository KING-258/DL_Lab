import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
X = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).reshape(-1, 1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).reshape(-1, 1)
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.fill_(1)
    model.bias.fill_(1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_list = []
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())    
    if(epoch % 50 == 0):
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, w: {model.weight.item():.4f}, b: {model.bias.item():.4f}')
plt.plot(range(epochs), loss_list, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs during Training')
plt.legend()
plt.grid(True)
plt.show()
print(f'\nFinal model parameters after training:\nWeight: {model.weight.item():.4f}, Bias: {model.bias.item():.4f}')