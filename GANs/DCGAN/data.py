from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

# define MNIST DataLoader
def get_MNIST_dataloader(root='./data/MNIST/', download=False, batch_size=128):
    # define transform
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # download/load MNIST dataset
    train_dataset = MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = MNIST(root=root, train=False, transform=transform)

    MNIST_dataloader = {
        'train': DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        ),
        'test': DataLoader(
            dataset=train_dataset,
            batch_size=len(train_dataset),
            shuffle=True
        )
    }

    return MNIST_dataloader