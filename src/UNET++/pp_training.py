import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from segmentation_ROIpp import UNETpp
import cv2
import os
import numpy as np
import csv
from datetime import datetime
from PIL import Image
import signal
import sys
import argparse

# Constants
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
TRAIN_IMG_DIR = "./data/train_images/"
TRAIN_MASK_DIR = "./data/train_masks/"
SAVE_CHECKPOINT_PATH = "./models/unetpp_checkpoint.pth.tar"
CSV_PATH = "./logs/training_logs.csv"

class LFADataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        
        # Extract the image number and construct the mask filename
        # From "image_123.jpg" to "image_123_mask.gif"
        image_name = self.images[index]
        image_number = image_name.split('.')[0]  # Get "image_123"
        mask_name = f"{image_number}_mask.gif"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # print(f"Loading mask from: {mask_path}")
        # print(f"File exists: {os.path.exists(mask_path)}")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            if mask is None:
                print(f"Warning: Mask loaded as None from {mask_path}")
        except Exception as e:
            print(f"Error loading mask {mask_path}: {str(e)}")
            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)  # Create empty mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)

def train_fn(loader, model, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    deep_losses = [0, 0, 0]  # Track losses for each deep supervision output
    
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(DEVICE)
        targets = targets.float().to(DEVICE)
        
        # Forward pass (get three outputs due to deep supervision)
        outputs = model(data)
        
        # Calculate loss for each output (deep supervision)
        losses = []
        for output in outputs:
            losses.append(loss_fn(output, targets))
        
        # Weighted sum of losses (you can adjust weights)
        weights = [0.3, 0.3, 0.4]  # More weight to final output
        loss = sum(w * l for w, l in zip(weights, losses))
        
        # Update running losses
        total_loss += loss.item()
        for i, l in enumerate(losses):
            deep_losses[i] += l.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate average losses
    avg_loss = total_loss / len(loader)
    avg_deep_losses = [l / len(loader) for l in deep_losses]
    
    return avg_loss, avg_deep_losses

def save_checkpoint(state, filename=SAVE_CHECKPOINT_PATH):
    print("=> Saving checkpoint")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def log_metrics(epoch, avg_loss, deep_losses, csv_path):
    metrics = {
        'epoch': epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'average_loss': avg_loss,
        'output1_loss': deep_losses[0],
        'output2_loss': deep_losses[1],
        'output3_loss': deep_losses[2]
    }
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
    
    # Append metrics
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writerow(metrics)

def signal_handler(sig, frame):
    print('\nGracefully shutting down...')
    # Save checkpoint before exiting
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    save_checkpoint(checkpoint)
    sys.exit(0)

def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train UNET++ model')
    parser.add_argument('--checkpoint', type=str, help='Path to custom checkpoint file to resume training from')
    parser.add_argument('-r', '--reset', action='store_true', help='Start training from scratch, ignoring any existing checkpoint')
    args = parser.parse_args()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = UNETpp(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = LFADataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=PIN_MEMORY,
    )

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(SAVE_CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    print(f"Starting training on device: {DEVICE}")
    print(f"Training logs will be saved to: {CSV_PATH}")
    
    # Load checkpoint if specified or if default checkpoint exists
    start_epoch = 0
    if not args.reset:
        if args.checkpoint:
            start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
        elif os.path.exists(SAVE_CHECKPOINT_PATH):
            start_epoch = load_checkpoint(SAVE_CHECKPOINT_PATH, model, optimizer)
    
    if start_epoch == 0:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            avg_loss, deep_losses = train_fn(train_loader, model, optimizer, loss_fn, epoch)
            
            # Log metrics
            log_metrics(epoch + 1, avg_loss, deep_losses, CSV_PATH)
            
            # Print progress
            print(f"Epoch: {epoch+1}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Deep Supervision Losses: {[f'{l:.4f}' for l in deep_losses]}")
            print("-" * 50)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                save_checkpoint(checkpoint)
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        # Save checkpoint before exiting
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()
