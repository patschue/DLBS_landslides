from torchmetrics.classification import MulticlassF1Score

# Möglicher Fehler: Loss durch Anzahl Bilder dividieren?

def train_deeplab(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.squeeze(1)
        y = y.long()  

        # Compute prediction error
        pred = model(X)['out']
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches

    return train_loss

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