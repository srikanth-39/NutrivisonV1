import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from config import BEST_MODEL
from dataloader import get_dataloaders
from model import build_model


def evaluate():
    # -------------------------
    # Device
    # -------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------
    # Load Data
    # -------------------------
    _, _, test_loader, class_names, _ = get_dataloaders()
    num_classes = len(class_names)

    # -------------------------
    # Load Checkpoint
    # -------------------------
    checkpoint = torch.load(
        BEST_MODEL,
        map_location=device
    )

    # -------------------------
    # Build Model
    # -------------------------
    model = build_model(num_classes)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    # -------------------------
    # Store Results
    # -------------------------
    y_true = []
    y_pred = []

    # -------------------------
    # Evaluation
    # -------------------------
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # -------------------------
    # Metrics
    # -------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate()