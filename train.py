import torch
import torch.nn as nn
import torch.optim as optim

from config import NUM_EPOCHS, BEST_MODEL
from dataloader import get_dataloaders
from model import build_model
from engine import train_one_epoch, validate_one_epoch


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names, _ = get_dataloaders()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW with weight decay and a safe fine-tuning learning rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=0.01
    )

    # Reduce learning rate when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes
            }, BEST_MODEL)
            print("--> Saved New Best Model Checkpoint")


if __name__ == "__main__":
    main()