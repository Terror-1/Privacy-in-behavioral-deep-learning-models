import torch 
import opacus
from opacus import PrivacyEngine
import pandas as pd

EMAKI_PATH = "../datasets/sample.pkl"
dataset = pd.read_pickle(EMAKI_PATH)
print(dataset)
model = torch.nn.Sequential(
    torch.nn.Linear(2, 2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid()
)


optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

privacy_engine = PrivacyEngine()
"""model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
) """
