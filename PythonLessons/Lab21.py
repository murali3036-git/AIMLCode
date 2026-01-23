import torch
import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# step 1 :- class which wraps the nn 
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# step 2 :- loss of accuracy
loss_fn = nn.MSELoss()
# step 3 :- optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
# step 4 :- run he model wiht lot tests , checks the error
# for checking this uses the optimizer
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

test_x = torch.tensor([[5.0]])
print("Prediction for x = 5 :", model(test_x).item())

# step 5 :- save the current state in to file.
torch.save(model.state_dict(), "linear_model.pth")

loaded_model = LinearModel()
loaded_model.load_state_dict(torch.load("linear_model.pth"))
loaded_model.eval()

test_x2 = torch.tensor([[6.0]])
print("Prediction for x = 6 (loaded model):", loaded_model(test_x2).item())
