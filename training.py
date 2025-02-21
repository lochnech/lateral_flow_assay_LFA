import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # for progress
import torch.nn as nn
import torch.optim as optim
from segmentation_ROI import UNET
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_image, )

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 2000
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True  # for further epochs when i have already saved training then we can turn it to true


def train_fn(loader, model, optimizer, loss_fn, scaler):
    train_loss = 0.0
    current_lr = optimizer.param_groups[0]["lr"]
    loop = tqdm(loader)
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

def main():
    train_transform = A.Compose(
        [
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
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # as the mask is binary so i used out as 1
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy without sigmoid in the outer layer
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

    if LOAD_MODEL:
        load_checkpoint(torch.load("models/model_checkpoint.pth.tar", map_location=DEVICE), model)
        _, val_accuracy = check_accuracy(val_loader, model, device=DEVICE)
        accuracy = max(val_accuracy, accuracy)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Epoch {epoch}: {train_loss}")

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        # check accuracy
        val_loss = check_accuracy(val_loader, model, device=DEVICE)
        scheduler.step(val_loss)
        # print some examples to a fold
        save_predictions_as_image(val_loader, model, folder="./saved_images/", device=DEVICE)


def predict_roi(image_path, model_path):
    # Load the trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Match your training size
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
