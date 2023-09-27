import matplotlib.pyplot as plt
import torch
import wandb

def plot_after_training_deeplab(test_loader, model, device):
    # Angenommen, test_loader ist dein DataLoader für die Testdaten
    test_images, test_masks = next(iter(test_loader))
    test_images, test_masks = test_images.to(device), test_masks.to(device)

    # Vorhersagen treffen
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    with torch.no_grad():
        predictions = model(test_images)['out']
        _, preds = torch.max(predictions, dim=1)  # Finde die Klasse mit der höchsten Wahrscheinlichkeit

    # Visualisiere das Bild, die echte Maske und die Vorhersage
    fig, axs = plt.subplots(10, 3, figsize=(15, 40))  # 10 Reihen, 3 Spalten für 10 Bilder

    for i in range(10):  # 10 Bilder
        axs[i, 0].imshow(test_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')  # Beachten Sie, dass wir die Dimensionen ändern, um das Bild zu zeigen
        # axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Beispiel {i + 1}: Bild')

        axs[i, 1].imshow(test_masks[i].squeeze(0).cpu(), cmap='gray')
        # axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Beispiel {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        # axs[i, 2].axis('off')
        axs[i, 2].set_title(f'Beispiel {i + 1}: Vorhersage')

    plt.tight_layout()
    wandb.log({"chart": wandb.Image(plt)})
    plt.show()


def plot_after_training_unet(test_loader, model, device):
    # Angenommen, test_loader ist dein DataLoader für die Testdaten
    test_images, test_masks = next(iter(test_loader))
    test_images, test_masks = test_images.to(device), test_masks.to(device)

    # Vorhersagen treffen
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    with torch.no_grad():
        predictions = model(test_images)
        _, preds = torch.max(predictions, dim=1)  # Finde die Klasse mit der höchsten Wahrscheinlichkeit

    # Visualisiere das Bild, die echte Maske und die Vorhersage
    fig, axs = plt.subplots(10, 3, figsize=(15, 40))  # 10 Reihen, 3 Spalten für 10 Bilder

    for i in range(10):  # 10 Bilder
        axs[i, 0].imshow(test_images[i].permute(1,2,0).cpu().numpy(), cmap='gray')  # Beachten Sie, dass wir die Dimensionen ändern, um das Bild zu zeigen
        # axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Beispiel {i + 1}: Bild')

        axs[i, 1].imshow(test_masks[i].squeeze(0).cpu(), cmap='gray')
        # axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Beispiel {i + 1}: Echte Groundtruth')

        axs[i, 2].imshow(preds[i].cpu(), cmap='gray')
        # axs[i, 2].axis('off')
        axs[i, 2].set_title(f'Beispiel {i + 1}: Vorhersage')

    plt.tight_layout()
    wandb.log({"chart": wandb.Image(plt)})
    plt.show()