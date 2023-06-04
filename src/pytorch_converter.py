import torch 
import opacus
import torch.nn as nn
from opacus import PrivacyEngine
import pandas as pd
from EMAKI_DataLoader import CustomEMAKIDataset
from statistics import mean



EMAKI_PATH = "../datasets/sample.pkl"
dataset = CustomEMAKIDataset()
#print(dataset)
model = torch.nn.Sequential(
    torch.nn.Linear(15, 5),
    torch.nn.Sigmoid(),
    torch.nn.Linear(5, 1),
    torch.nn.Sigmoid()
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
) 

criterion = nn.CrossEntropyLoss()
epochs = 10
for epoch in range(epochs):
    accs = []
    losses = []
    for data, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(-1)
        n_correct = float(preds.eq(labels).sum())
        batch_accuracy = n_correct / len(labels)
        accs.append(batch_accuracy)
        losses.append(float(loss))
    printstr = (f"\t Epoch {epoch}. Accuracy: {mean(accs):.6f} | Loss: {mean(losses):.6f}" )
    print(printstr)