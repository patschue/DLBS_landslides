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
    train_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y = y.squeeze(1)
        y = y.long()  

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches

    return train_loss