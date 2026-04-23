import os
import json
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset

class DonutSROIEDataset(Dataset):
    def __init__(self, img_dir, ent_dir, processor, max_length=768, split="train"):
        """
        Custom Dataset for SROIE receipt parsing with Donut model.
        Args:
            img_dir: Directory containing receipt images.
            ent_dir: Directory containing ground truth text files (JSON or key:value format).
            processor: DonutProcessor for image and text tokenization.
            max_length: Maximum token sequence length for the generator.
        """
        self.img_dir = img_dir
        self.ent_dir = ent_dir
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Heavy augmentations for real-world scanned receipts (noise, rotation, perspective)
        self.transform_real = A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.7),
                A.Perspective(scale=(0.02, 0.05), p=0.6),
            ], p=0.8),
            A.OneOf([
                A.Sharpen(alpha=(0.1, 0.3), p=0.5),
                A.ImageCompression(quality_lower=70, quality_upper=90, p=0.4),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.6),
                A.Emboss(p=0.3),
            ], p=0.7),
            A.CoarseDropout(
                max_holes=10, 
                max_height=0.02, 
                max_width=0.04, 
                min_holes=5, 
                fill_value=255, 
                p=0.4
            ),
            A.ToGray(p=0.15),
        ])

        # Light augmentations for synthetic or clean digital images
        self.transform_synth = A.Compose([
            A.Perspective(scale=(0.01, 0.03), p=0.4), 
            A.CLAHE(clip_limit=2.0, p=0.5),           
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
            A.ToGray(p=0.1),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_id = os.path.splitext(filename)[0]
        image = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        
        # Apply data augmentation only during training phase
        if self.split == "train":
            image_np = np.array(image)
            
            # Use specific transforms for synthetic/generated data
            if "ultra" in filename.lower() or "synth" in filename.lower():
                augmented = self.transform_synth(image=image_np)
            else:
                augmented = self.transform_real(image=image_np)
                
            image = Image.fromarray(augmented['image'])
        
        # Format the target text sequence using special Donut prompt tokens
        target_sequence = "<s_sroie>"
        ent_path = os.path.join(self.ent_dir, f"{file_id}.txt")
        data = {}

        # Parsing ground truth files (supports both JSON and raw colon-separated text)
        if os.path.exists(ent_path):
            with open(ent_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    for line in content.split('\n'):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            data[k.strip().lower()] = v.strip()

            # Construct the ground truth string: <s_key>value</s_key>
            for key in ["company", "date", "address", "total", "cash", "change"]:
                value = data.get(key, "")
                target_sequence += f"<s_{key}>{value}</s_{key}>"
        
        target_sequence += "</s_sroie>"

        # Preprocess image into pixel values tensor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Tokenize the target text sequence
        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Set pad tokens to -100 so the loss function ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}