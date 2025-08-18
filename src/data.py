from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BATCH_SIZE, IMG_SIZE, NUM_WORKERS, TEST_DIR, TRAIN_DIR, VAL_DIR


def get_transforms():
    """
    Returns two variances of transormation:
    -train_tf - with augmentation, random changes
    -eval_tf - without augmentation -  reproduction and validation
    """
    train_tf = transforms.Compose(
        [
            # Teach the model to handle different crops/zooms like real phone photos
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def get_datasets():
    """Creates 3 sets of images (train/val/test) with adequate transformers."""
    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)
    return train_ds, val_ds, test_ds


def get_dataloaders():
    """
    Returns 3 dataloaders.
    - train: shuffle=True (mixed samples),
    - val/test: shuffle=False.
    """
    train_ds, val_ds, test_ds = get_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes (order = indicies):", classes)
    print("Batch number train/val/test:", len(train_loader), len(val_loader), len(test_loader))
    # images count
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"Samples: train={n_train}, val={n_val}, test={n_test}")
