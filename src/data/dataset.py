import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class XViewDataset(Dataset):
    """
    Custom PyTorch Dataset for xView-like data.
    Expects a DataFrame of bounding boxes plus a directory of images.
    """
    def __init__(self, df, images_dir, transforms=None):
        self.df = df
        self.images_dir = images_dir
        self.transforms = transforms

        # Unique image IDs
        self.image_ids = df["image_id"].unique().tolist()

        # Group bounding boxes by image_id
        self.records_by_id = {}
        for image_id in self.image_ids:
            sub_df = df[df["image_id"] == image_id]
            self.records_by_id[image_id] = sub_df.to_dict("records")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Collect bounding boxes + labels
        records = self.records_by_id[image_id]
        boxes, labels = [], []
        for r in records:
            boxes.append([r["x_min"], r["y_min"], r["x_max"], r["y_max"]])
            labels.append(r["class_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target, image_id

def collate_fn(batch):
    """
    DataLoader collate function for object detection.
    """
    return tuple(zip(*batch))

def get_transforms():
    """
    Returns torchvision transforms for training/validation.
    Extend as needed for augmentation.
    """
    from torchvision import transforms
    return transforms.Compose([transforms.ToTensor()])