from torchmetrics.classification import MulticlassF1Score
import torch


def validation_deeplab(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, f1_value = 0, 0

    f1_score = MulticlassF1Score(num_classes=2, average='none').to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.squeeze(1)
            y = y.long()  

            pred = model(X)['out']
            test_loss += loss_fn(pred, y).item()

            f1_value += f1_score(pred, y)


    test_loss /= num_batches
    f1_value /= num_batches

    f1_value_background, f1_value_landslide = f1_value.tolist()

    return test_loss, f1_value_background, f1_value_landslide

def validation_unet(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, f1_value = 0, 0

    f1_score = MulticlassF1Score(num_classes=2, average='none').to(device)

    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y = y.squeeze(1)
            y = y.long()  

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            f1_value += f1_score(pred, y)


    test_loss /= num_batches
    f1_value /= num_batches

    f1_value_background, f1_value_landslide = f1_value.tolist()

    return test_loss, f1_value_background, f1_value_landslide