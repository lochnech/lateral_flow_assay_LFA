import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.UNET.segmentation_ROI import UNET
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import csv
from datetime import datetime
import signal
import sys
import argparse

from src.utils.utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_image)

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 2000
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
SAVE_CHECKPOINT_PATH = "models/model_checkpoint.pth.tar"
CSV_PATH = "logs/training_logs.csv"

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    model.train()
    train_loss = 0.0
    current_lr = optimizer.param_groups[0]["lr"]
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_index, (data, targets) in enumerate(loop):
        torch.cuda.empty_cache()
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(
            loss=loss.item(),
            lr=current_lr
        )
        train_loss += loss.item()

    return train_loss / len(loader)

def log_metrics(epoch, train_loss, val_loss, val_accuracy, csv_path):
    metrics = {
        'epoch': epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'learning_rate': optimizer.param_groups[0]["lr"]
    }
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
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
    save_checkpoint(checkpoint, SAVE_CHECKPOINT_PATH)
    sys.exit(0)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train UNET model')
    parser.add_argument('--checkpoint', type=str, help='Path to custom checkpoint file to resume training from')
    parser.add_argument('-r', '--reset', action='store_true', help='Start training from scratch, ignoring any existing checkpoint')
    args = parser.parse_args()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    train_loader, val_loader = get_loaders(
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(SAVE_CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    print(f"Starting training on device: {DEVICE}")
    print(f"Training logs will be saved to: {CSV_PATH}")
    
    # Load checkpoint if specified or if default checkpoint exists
    start_epoch = 0
    best_val_accuracy = 0
    
    if not args.reset:
        if args.checkpoint:
            start_epoch = load_checkpoint(torch.load(args.checkpoint, map_location=DEVICE), model)
        elif os.path.exists(SAVE_CHECKPOINT_PATH):
            start_epoch = load_checkpoint(torch.load(SAVE_CHECKPOINT_PATH, map_location=DEVICE), model)
    
    if start_epoch == 0:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            # Train
            train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)
            
            # Validate
            val_loss, val_accuracy = check_accuracy(val_loader, model, device=DEVICE)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            log_metrics(epoch + 1, train_loss, val_loss, val_accuracy, CSV_PATH)
            
            # Print progress
            print(f"\nEpoch: {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy
                }
                save_checkpoint(checkpoint, SAVE_CHECKPOINT_PATH)
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                save_checkpoint(checkpoint, SAVE_CHECKPOINT_PATH)
            
            # Save predictions
            save_predictions_as_image(val_loader, model, folder="./saved_images/", device=DEVICE)
            
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        # Save checkpoint before exiting
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(checkpoint, SAVE_CHECKPOINT_PATH)

def predict_roi(image_path, model_path):
    # Load the trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        prediction = torch.sigmoid(model(image_tensor))
        mask = (prediction > 0.5).float()
        
    return mask.cpu().numpy()[0, 0]  # Return binary mask

if __name__ == "__main__":
    main()
