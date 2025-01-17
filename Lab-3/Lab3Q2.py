import torch
x = torch.tensor([2.0, 3.0])
y = torch.tensor([20.0, 40.0])
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)
lr = 0.001
epochs = 25
for epoch in range(epochs):
    y_pred = w * x + b
    diff = y_pred - y
    loss = torch.mean(diff ** 2) / 2
    w.grad = None
    b.grad = None
    loss.backward()
    print(f"Epoch {epoch+1}:")
    print(f"w.grad = {w.grad.item():.4f}, b.grad = {b.grad.item():.4f}")
    w.data -= lr * w.grad
    b.data -= lr * b.grad
    print(f"Updated w = {w.item():.4f}, Updated b = {b.item():.4f}")
    print("-" * 40)