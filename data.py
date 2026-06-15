import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10(batch_size=128, num_workers=2, data_dir='./data'):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


CIFAR10_META = {
    'num_classes': 10,
    'height': 32,
    'width': 32,
    'channels': 3,
    'classes': ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck'],
}

if __name__ == '__main__':
    train_loader, val_loader = get_cifar10()
    images, labels = next(iter(train_loader))
    print(f'Batch shape : {images.shape}')   # (128, 3, 32, 32)
    print(f'Label shape : {labels.shape}')   # (128,)
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches  : {len(val_loader)}')