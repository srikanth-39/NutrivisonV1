from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE
from transforms import train_transform, val_transform, test_transform

def get_dataloaders(batch_size=BATCH_SIZE):
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    test_dataset = ImageFolder(root=TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    class_idx = train_dataset.class_to_idx

    return train_loader, val_loader, test_loader, class_names, class_idx