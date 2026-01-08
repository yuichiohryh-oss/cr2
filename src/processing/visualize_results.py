import sys
import os
# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src', 'training')) # For direct imports in train_model1.py

import torch
import cv2
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from src.training.train_model1 import ActionRecognitionModel, NUM_CLASSES
from src.training.dataset import ClashRoyaleDataset

# Config
MODEL_PATH = "model1.pth"
CSV_FILE = "dataset/labeled_actions.csv"
IMAGE_DIR = "images"
OUTPUT_DIR = "verification_results"
LABELS = {
    0: "Skeleton",
    1: "Evo Skeleton",
    2: "Ice Spirit",
    3: "Ice Golem",
    4: "Hog Rider",
    5: "Musketeer",
    6: "Fireball",
    7: "Cannon",
    8: "Evo Cannon",
    9: "The Log"
}

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = ActionRecognitionModel(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Dataset
    # We want standard transforms but no augmentation for viz usually, 
    # but the training transform was basic resize/norm anyway.
    # We need a transform for the model, and raw image for drawing.
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ClashRoyaleDataset(csv_file=CSV_FILE, root_dir=IMAGE_DIR, transform=transform)
    
    print(f"Visualizing {len(dataset)} samples...")
    
    # Iterate
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Input for model
        img_tensor = sample['image'].unsqueeze(0).to(device) # (1, C, H, W)
        
        with torch.no_grad():
            pred_cls, pred_reg = model(img_tensor)
            
        # Decode prediction
        pred_label_idx = torch.argmax(pred_cls, dim=1).item()
        pred_x, pred_y = pred_reg[0].cpu().numpy()
        
        # Ground Truth
        gt_label_idx = sample['unit_id'].item()
        gt_x, gt_y = sample['coordinate'].numpy()
        
        # Load Raw Image for drawing (Re-load to avoid tensor conversion artifacts)
        # Using internal helper from dataset would be best, but it's private.
        # We can implement a simple 'get_raw_image' logic or just trust the dataset loader
        # But dataset resizes to 224x224. We want to draw on original or at least visible size.
        
        # Let's peek at the dataset implementation... 
        # Actually dataset.__getitem__ returns a tensor. 
        # Let's just create a raw list to match.
        
        # Retrieve timestamp to find original file
        row = dataset.annotations.iloc[i]
        timestamp = row['timestamp']
        
        # Simple find closest again (copy logic for independence)
        # Or better: make dataset return path? No, standard dataset doesn't.
        # Let's assumes dataset works 1:1 with rows.
        
        # Logic to find file
        closest_file = None
        min_diff = 1.0
        # We need the file list... expensive to glob every time.
        # Let's just trust dataset returns consistent index.
        # Wait, I can't easily get the filename from the dataset object standardly.
        
        # Hack: dataset.dataset is not available?
        # Let's just do the search here once.
        import glob
        all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")), key=lambda x: float(os.path.basename(x).split(".jpg")[0]))
        
        # Binary search or linear? Linear is fine for 100 items.
        # But wait, dataset logic is: find_closest_image(timestamp).
        
        target_ts = timestamp
        best_file = None
        local_min = 1.0
        for img_path in all_images:
             ts = float(os.path.basename(img_path).replace(".jpg", ""))
             diff = abs(ts - target_ts)
             if diff < local_min:
                 local_min = diff
                 best_file = img_path
        
        if best_file:
            img_vis = cv2.imread(best_file)
        else:
            img_vis = np.zeros((900, 500, 3), dtype=np.uint8) # Dummy
            
        # Draw on img_vis
        h, w, _ = img_vis.shape
        
        # Colors
        color_gt = (0, 255, 0) # Green
        color_pred = (0, 0, 255) # Red
        
        # Draw Points
        pos_gt = (int(gt_x * w), int(gt_y * h))
        pos_pred = (int(pred_x * w), int(pred_y * h))
        
        cv2.circle(img_vis, pos_gt, 10, color_gt, -1)
        cv2.circle(img_vis, pos_pred, 8, color_pred, 2)
        
        # Labels
        name_gt = LABELS.get(gt_label_idx, f"Unk:{gt_label_idx}")
        name_pred = LABELS.get(pred_label_idx, f"Unk:{pred_label_idx}")
        
        text_gt = f"GT: {name_gt}"
        text_pred = f"Pred: {name_pred} ({pos_pred[0]},{pos_pred[1]})"
        
        cv2.putText(img_vis, text_gt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_gt, 2)
        cv2.putText(img_vis, text_pred, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_pred, 2)
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, f"result_{i:03d}_{name_gt}.jpg")
        cv2.imwrite(out_path, img_vis)
        
    print(f"Saved {len(dataset)} visualization images to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
