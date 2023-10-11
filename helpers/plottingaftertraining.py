import matplotlib.pyplot as plt
import torch
import wandb

def plot_samples_deeplab(loader, model, device, title_suffix):
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images)['out']
        _, preds = torch.max(predictions, dim=1)

    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    for i in range(10):
        axs[i, 0].imshow(images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f'Beispiel {i + 1}: Bild')

        axs[i, 1].imshow(masks[i].squeeze(0).cpu(), cmap='gray')
        axs[i, 1].set_title(f'Beispiel {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        axs[i, 2].set_title(f'Beispiel {i + 1}: Vorhersage {title_suffix}')

    plt.tight_layout()
    wandb.log({f"chart_{title_suffix}": wandb.Image(plt)})
    plt.show()

def plot_samples_unet(loader, model, device, title_suffix):
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images)
        _, preds = torch.max(predictions, dim=1)

    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    for i in range(10):
        axs[i, 0].imshow(images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f'Beispiel {i + 1}: Bild')

        axs[i, 1].imshow(masks[i].squeeze(0).cpu(), cmap='gray')
        axs[i, 1].set_title(f'Beispiel {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        axs[i, 2].set_title(f'Beispiel {i + 1}: Vorhersage {title_suffix}')

    plt.tight_layout()
    wandb.log({f"chart_{title_suffix}": wandb.Image(plt)})
    plt.show()