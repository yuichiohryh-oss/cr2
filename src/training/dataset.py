import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np

class ClashRoyaleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        # Filter out unlabeled (-1)
        self.annotations = self.annotations[self.annotations['unit_id'] != -1].reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Determine image size from first image to normalize coordinates if needed
        # For now, we assume we return raw or normalized floats
        # The labels in CSV are pixels.
        # We need the full screenshot corresponding to the 'Action'. 
        # Wait, the CSV has 'card_image' (crop) but what is the input to Model 1?
        # Model 1 input: The GAME SCREEN at the time of placement (or slightly before).
        
        # The 'dataset/labeled_actions.csv' has:
        # timestamp, card_image, deck_x, deck_y, arena_x, arena_y, action_type, unit_id
        
        # We need to find the SCREENSHOT corresponding to 'timestamp'.
        # The 'card_image' filename is derived from deck click timestamp, not arena click.
        # But 'timestamp' column is the arena click timestamp.
        # We need to find the closest full screenshot in 'images/' to this timestamp.
        
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.jpg')])
        self.image_timestamps = np.array([float(f.replace('.jpg', '')) for f in self.image_files])
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations.iloc[idx]
        action_ts = row['timestamp']
        
        # Find closest image
        # searching every time is slow, but fine for small dataset
        diffs = np.abs(self.image_timestamps - action_ts)
        min_idx = np.argmin(diffs)
        img_name = self.image_files[min_idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, _ = image.shape
        
        # Labels
        unit_id = int(row['unit_id'])
        # Normalize coordinates 0-1
        target_x = row['arena_x'] / w
        target_y = row['arena_y'] / h
        
        # Basic Transform if none provided
        if self.transform:
            image = self.transform(image)
        else:
            # To Tensor
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            
        return {
            'image': image,
            'unit_id': torch.tensor(unit_id, dtype=torch.long),
            'coordinate': torch.tensor([target_x, target_y], dtype=torch.float32)
        }
