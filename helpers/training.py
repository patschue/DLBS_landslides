from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import BinaryF1Score
import torch

def train_deeplab(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, f1_value = 0, 0
    model.train()

    f1_score = MulticlassF1Score(num_classes=2, average='none').to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.squeeze(1)
        y = y.long()  

        # Compute prediction error
        pred = model(X)['out']
        print(pred)
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()

        f1_value += f1_score(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    f1_value /= num_batches

    f1_value_background, f1_value_landslide = f1_value.tolist()
    return train_loss, f1_value_background, f1_value_landslide



def train_deeplab_binary(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, f1_value = 0, 0
    model.train()

    f1_score_landslide = BinaryF1Score().to(device)
    f1_score_background = BinaryF1Score().to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Ändern Sie das Label in float 
        y = y.float()


        # Compute prediction error
        pred = model(X)['out']
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # F1 Score für Landslide
        f1_score_landslide(torch.sigmoid(pred), y)

        # F1 Score für den Hintergrund
        f1_score_background(1 - torch.sigmoid(pred), 1 - y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches

    f1_value_background = f1_score_background.compute().item()
    f1_value_landslide = f1_score_landslide.compute().item()

    return train_loss, f1_value_background, f1_value_landslide



def train_unet(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, f1_value = 0, 0
    model.train()

    f1_score = MulticlassF1Score(num_classes=2, average='none').to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y = y.squeeze(1)
        y = y.long()  

        # Compute prediction error
        pred = model(X)
        print(pred)
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()

        f1_value += f1_score(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    f1_value /= num_batches

    f1_value_background, f1_value_landslide = f1_value.tolist()
    return train_loss, f1_value_background, f1_value_landslide


def train_unet_maxbatches(dataloader, model, loss_fn, optimizer, device, max_batches=1):
    size = len(dataloader.dataset)
    train_loss, f1_value = 0, 0
    model.train()
    
    f1_score = MulticlassF1Score(num_classes=2, average='none').to(device)

    for batch, (X, y) in enumerate(dataloader):
        if batch >= max_batches:  # Begrenzung auf die gewünschte Anzahl von Batches
            break
        
        X, y = X.to(device), y.to(device)
        
        y = y.squeeze(1)
        y = y.long()  

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        f1_value += f1_score(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= max_batches

    f1_value /= max_batches

    f1_value_background, f1_value_landslide = f1_value.tolist()

    return train_loss, f1_value_background, f1_value_landslide

def train_unet_binary(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, f1_value = 0, 0
    model.train()

    f1_score_landslide = BinaryF1Score().to(device)
    f1_score_background = BinaryF1Score().to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y = y.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        print(pred)
        pred_labels = (torch.sigmoid(pred) > 0.5).float()
        print(pred_labels)
        # F1 Score für Landslide
        f1_score_landslide(pred_labels, y)

        # F1 Score für den Hintergrund
        f1_score_background(1 - pred_labels, 1 - y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches

    f1_value_background = f1_score_background.compute().item()
    f1_value_landslide = f1_score_landslide.compute().item()

    return train_loss, f1_value_background, f1_value_landslide
