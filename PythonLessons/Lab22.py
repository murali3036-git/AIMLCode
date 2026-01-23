import pandas as pd
import torch
import torch.nn as nn

data = pd.read_csv("taxi_fare_data.csv")
x = torch.tensor(
    data[["distance", "time", "traffic", "night"]].values,
    dtype=torch.float32
)

y = torch.tensor(
    data[["fare"]].values,
    dtype=torch.float32
)

class TaxiFareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)  # first linear layer
        self.relu = nn.ReLU()        # non-linearity
        self.fc2 = nn.Linear(10, 1)  # output fare

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TaxiFareModel()


loss_fn = nn.MSELoss()  # mean squared error for regression

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 2000
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

test_input = torch.tensor([[1.0,15.0,1.0,1.0]])  # Example ride
predicted_fare = model(test_input).item()
print("Predicted fare:", predicted_fare)

torch.save(model.state_dict(), "taxi_fare_model.pth")

loaded_model = TaxiFareModel()
loaded_model.load_state_dict(torch.load("taxi_fare_model.pth"))
loaded_model.eval()

test_input2 = torch.tensor([[1.0,15.0,1.0,1.0]])
predicted_fare2 = loaded_model(test_input2).item()
print("Predicted fare (loaded model):", predicted_fare2)
