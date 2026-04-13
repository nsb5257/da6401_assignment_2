"""Dataset skeleton for Oxford-IIIT Pet."""

import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    def __init__(self, split: str = "train", data_dir: str = "./data"):
        """
        Initialize the multi-task dataset.
        Args:
            split: 'train' or 'test'.
            data_dir: Base directory containing the dataset.
        """
        self.split = split
        self.data_dir = data_dir
        
        # --- Augmentation Pipeline using Albumentations ---
        if self.split == "train":
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.image_paths = [] 
        self.masks_paths = [] 
        self.bboxes = []      
        self.labels = []      

        # --- ACTUAL DATASET PARSING LOGIC ---
        images_dir = os.path.join(data_dir, "images")
        masks_dir = os.path.join(data_dir, "annotations", "trimaps")
        xml_dir = os.path.join(data_dir, "annotations", "xmls")
        
        # The dataset provides a split file: trainval.txt or test.txt
        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(data_dir, "annotations", split_file)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"\nError:The dataset is looking for the file here:\n{os.path.abspath(split_path)}\nBut the file does not exist at this location!")
        with open(split_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            image_name = parts[0]
            # Breed ID is 1-indexed in the dataset, we convert to 0-indexed (0 to 36)
            class_id = int(parts[1]) - 1 

            img_path = os.path.join(images_dir, f"{image_name}.jpg")
            mask_path = os.path.join(masks_dir, f"{image_name}.png")
            xml_path = os.path.join(xml_dir, f"{image_name}.xml")

            # ONLY add to dataset if the image, mask, AND bounding box XML exist
            if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(xml_path):
                try:
                    # Parse XML to get bounding box
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find('size')
                    img_width = float(size.find('width').text)
                    img_height = float(size.find('height').text)
                    
                    bndbox = root.find('object').find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # Convert to Albumentations YOLO format (normalized center x, center y, w, h)
                    w = (xmax - xmin) / img_width
                    h = (ymax - ymin) / img_height
                    x_center = (xmin + xmax) / 2.0 / img_width
                    y_center = (ymin + ymax) / 2.0 / img_height
                    
                    # Clamp between 0 and 1 just in case of annotation errors
                    yolo_bbox = [
                        max(0, min(1, x_center)), 
                        max(0, min(1, y_center)), 
                        max(0, min(1, w)), 
                        max(0, min(1, h))
                    ]

                    # Append valid data
                    self.image_paths.append(img_path)
                    self.masks_paths.append(mask_path)
                    self.labels.append(class_id)
                    self.bboxes.append(yolo_bbox)
                except Exception as e:
                    # Skip any corrupted XML files silently
                    continue

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Fetches and augments a single multi-task sample."""
        img_path = self.image_paths[idx]
        mask_path = self.masks_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Shift trimaps from (1, 2, 3) to (0, 1, 2) for PyTorch
        mask_np = mask_np - 1 

        bbox = self.bboxes[idx]
        label = self.labels[idx]

        # Apply Synchronized Augmentations
        transformed = self.transform(
            image=image_np,
            mask=mask_np,
            bboxes=[bbox],
            class_labels=[label]
        )

        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        # Albumentations might drop a box if it gets cropped entirely out of frame; fallback if empty
        transformed_bbox = transformed['bboxes'][0] if len(transformed['bboxes']) > 0 else bbox

        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(transformed_image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(transformed_mask).long()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Bounding Box: Convert YOLO back to Pixel Space for the 224x224 image
        pixel_bbox = [
            transformed_bbox[0] * 224.0,
            transformed_bbox[1] * 224.0,
            transformed_bbox[2] * 224.0,
            transformed_bbox[3] * 224.0
        ]
        bbox_tensor = torch.tensor(pixel_bbox, dtype=torch.float32)

        return {
            'image': image_tensor,
            'label': label_tensor,
            'bbox': bbox_tensor,
            'mask': mask_tensor
        }