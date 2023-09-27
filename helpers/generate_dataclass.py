import os
from torchvision import transforms
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
import random
from PIL import Image


class SegmentationDataset(VisionDataset):
    def __init__(self, root, split='train', transform_mode='to_tensor'):
        super(SegmentationDataset, self).__init__(root)
        
        assert split in ['train', 'test', 'validation']
        self.split = split
        self.transform_mode = transform_mode
        
        # Ordnerpfade für Bilder und Masken
        self.images_dir = os.path.join(root, split, 'images')
        self.masks_dir = os.path.join(root, split, 'masks')
        
        # Liste der Dateinamen
        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))
        
        assert len(self.images) == len(self.masks)

    def transform_to_tensor(self, image, mask):
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)
        return image, mask

    def transform_normalize(self, image, mask):
        to_tensor = transforms.ToTensor()
        # Für die Normalisierung benötigen Sie die mittleren und std-Werte Ihrer Daten.
        # Hier nehme ich an, dass Sie mit den Standardwerten für Imagenet arbeiten.
        normalize = transforms.Normalize(mean=[0,0,0], std=[1,1,1])
        image = normalize(to_tensor(image))
        mask = to_tensor(mask)  # Masken normalisieren wir typischerweise nicht
        return image, mask

    def transform_flip(self, image, mask):
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0,0,0], std=[1,1,1])

        if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

        image = normalize(to_tensor(image))
        mask = to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        # Lade das Bild und die entsprechende Maske
        img_path = os.path.join(self.images_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.masks[index])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")
             
        if self.transform_mode == 'to_tensor':
            img, mask = self.transform_to_tensor(img, mask)
        elif self.transform_mode == 'normalize':
            img, mask = self.transform_normalize(img, mask)
        elif self.transform_mode == 'flip':
            img, mask = self.transform_flip(img, mask)

        return img, mask

    def __len__(self):
        return len(self.images)
