from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import BinaryF1Score
import torch

def validation_binary(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, f1_value = 0, 0

    f1_score_landslide = BinaryF1Score().to(device)
    f1_score_background = BinaryF1Score().to(device)
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y = y.float()

            pred = model(X)['out']
            val_loss += loss_fn(pred, y).item()

            f1_score_landslide(torch.sigmoid(pred), y)

            f1_score_background(1 - torch.sigmoid(pred), 1 - y)

    val_loss /= num_batches

    f1_value_background = f1_score_background.compute().item()
    f1_value_landslide = f1_score_landslide.compute().item()

    return val_loss, f1_value_background, f1_value_landslide