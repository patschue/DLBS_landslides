import matplotlib.pyplot as plt
import torch
import wandb

def plot_after_training_deeplab(test_loader, model, device):
    test_images, test_masks = next(iter(test_loader))
    test_images, test_masks = test_images.to(device), test_masks.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(test_images)['out']
        _, preds = torch.max(predictions, dim=1)

    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    for i in range(10):
        axs[i, 0].imshow(test_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f'Beispiel {i + 1}: Bild')

        axs[i, 1].imshow(test_masks[i].squeeze(0).cpu(), cmap='gray')
        axs[i, 1].set_title(f'Beispiel {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        axs[i, 2].set_title(f'Beispiel {i + 1}: Vorhersage')

    plt.tight_layout()
    wandb.log({"chart": wandb.Image(plt)})
    plt.show()


def plot_after_training_unet(test_loader, model, device):
    test_images, test_masks = next(iter(test_loader))
    test_images, test_masks = test_images.to(device), test_masks.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(test_images)
        _, preds = torch.max(predictions, dim=1)

    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    for i in range(10):
        axs[i, 0].imshow(test_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f'Beispiel {i + 1}: Bild')

        axs[i, 1].imshow(test_masks[i].squeeze(0).cpu(), cmap='gray')
        axs[i, 1].set_title(f'Beispiel {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        axs[i, 2].set_title(f'Beispiel {i + 1}: Vorhersage')

    plt.tight_layout()
    wandb.log({"chart": wandb.Image(plt)})
    plt.show()

def plot_after_training_unet2(train_loader, model, device):
    test_images, test_masks = next(iter(train_loader))
    test_images, test_masks = test_images.to(device), test_masks.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(test_images)
        _, preds = torch.max(predictions, dim=1)

    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    for i in range(10):
        axs[i, 0].imshow(test_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f'Beispiel1 {i + 1}: Bild')

        axs[i, 1].imshow(test_masks[i].squeeze(0).cpu(), cmap='gray')
        axs[i, 1].set_title(f'Beispiel1 {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        axs[i, 2].set_title(f'Beispiel1 {i + 1}: Vorhersage')

    plt.tight_layout()
    wandb.log({"chart": wandb.Image(plt)})
    plt.show()