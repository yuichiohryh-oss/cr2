import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
import copy
from dataset import ClashRoyaleDataset

# Configuration
DATA_DIR = "images"
CSV_FILE = "dataset/labeled_actions.csv"
BATCH_SIZE = 8 # Increase batch size slightly
EPOCHS = 50 
LR = 0.001
NUM_CLASSES = 10 # 0-9

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        # Load pre-trained ResNet
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        # Remove original FC
        self.backbone.fc = nn.Identity()
        
        # Classification Head
        self.cls_head = nn.Linear(num_ftrs, num_classes)
        
        # Regression Head (X, Y)
        self.reg_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 2), # x, y (0-1)
            nn.Sigmoid() # Force output 0-1
        )
        
    def forward(self, x):
        features = self.backbone(x)
        # ResNet global average pooling output is flattened? No, it's (B, 512, 1, 1). 
        # Identity replaces FC, but avgpool is before FC.
        # Check resnet structure... 
        # ResNet forward: x -> ... -> avgpool -> flatten -> fc
        # If we replace fc with Identity, we get (B, 512).
        
        cls_out = self.cls_head(features)
        reg_out = self.reg_head(features)
        return cls_out, reg_out

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # Force CPU for debugging
    print(f"Using device: {device}")
    
    # Transforms
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # ResNet standard
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    if not os.path.exists(CSV_FILE):
        print("Dataset CSV not found.")
        return

    dataset = ClashRoyaleDataset(csv_file=CSV_FILE, root_dir=DATA_DIR, transform=trans)
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Split (just train on all for now since it's tiny)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    model = ActionRecognitionModel(NUM_CLASSES).to(device)
    
    # Loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0
        
        for inputs in dataloader:
            imgs = inputs['image'].to(device)
            uids = inputs['unit_id'].to(device)
            coords = inputs['coordinate'].to(device)
            
            optimizer.zero_grad()
            
            pred_cls, pred_coords = model(imgs)
            
            loss_cls = criterion_cls(pred_cls, uids)
            loss_reg = criterion_reg(pred_coords, coords)
            
            # Weighted loss?
            loss = loss_cls + 10.0 * loss_reg # Weigh coord error heavily?
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
            # Acc
            _, preds = torch.max(pred_cls, 1)
            running_acc += torch.sum(preds == uids.data)
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_acc.double() / len(dataset)
        
        print(f"Epoch {epoch}/{EPOCHS-1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
    # Save
    torch.save(model.state_dict(), "model1.pth")
    print("Model saved to model1.pth")

if __name__ == "__main__":
    main()
