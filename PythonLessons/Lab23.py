import torch
import torch.nn as nn


class TaxiFareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



model = TaxiFareModel()
model.load_state_dict(torch.load("taxi_fare_model.pth"))
model.eval()



dummy_input = torch.randn(1, 4)

torch.onnx.export(
    model,
    dummy_input,
    "taxi_fare_model.onnx",
    input_names=["features"],
    output_names=["fare"]
    #opset_version=17
)

print("Model exported to taxi_fare_model.onnx")