from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, Caltech256,Caltech101
from torch.utils.data import DataLoader, Subset


from torch.utils.data import DataLoader, Subset,Dataset
import numpy as np
import random
from imutils import paths
import os
import cv2
from sklearn.model_selection import train_test_split


class FewShotSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = np.array(self.dataset.targets)[indices].tolist()


__all__ = [
    "cifar10_dataloaders",
    "cifar100_dataloaders",
    "svhn_dataloaders",
    "fashionmnist_dataloaders",
    "caltech101_dataloaders",
    "caltech256_dataloaders",
]


def sample_dataset(dataset, per):
    random.seed(1234)
    all_idxs = list()
    for i in range(10):
        idxs = np.where(np.array(dataset["targets"]) == i)[0].tolist()
        all_idxs += random.sample(idxs, 10)

    random.shuffle(all_idxs)

    dataset["targets"] = np.array(dataset["targets"])[all_idxs].tolist()
    dataset["data"] = np.array(dataset["data"])[all_idxs]
    return dataset


def get_balanced_subset(dataset, number_of_samples):
    unique_labels = np.unique(dataset.targets)
    train_idxs = []
    for label in unique_labels:
        number_of_samples_per_label = int(number_of_samples / len(unique_labels))

        idxs = np.where(np.array(dataset.targets) == label)[0].tolist()
        train_idxs += idxs[:number_of_samples_per_label]

    dataset_train = FewShotSubset(dataset, train_idxs)
    return dataset_train


def get_random_subset(dataset, number_of_samples):
    if number_of_samples > len(dataset):
        raise ValueError("number of samples is too large")
    idxs = np.random.choice(len(dataset), number_of_samples, replace=False)
    dataset_train = FewShotSubset(dataset, idxs)
    return dataset_train


def cifar10_dataloaders(
    batch_size=64,
    data_dir="datasets/cifar10",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
    balanced=False,
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        raise ValueError("subset ratio is not supported for cifar10")

    elif number_of_samples is not None:
        train_set = CIFAR10(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = CIFAR10(data_dir, train=True, transform=test_transform, download=True)
        val_set = Subset(val_set, list(range(40000, 50000)))
        train_set = Subset(train_set, list(range(40000)))
        if balanced:
            train_set, val_set = get_balanced_subset(
                train_set, number_of_samples
            )
        else:
            train_set = get_random_subset(train_set, number_of_samples)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=64,
    data_dir="datasets/cifar100",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
    balanced=False,
):

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        raise ValueError("subset ratio is not supported for cifar100")

    elif number_of_samples is not None:
        train_set = CIFAR100(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = CIFAR100(
            data_dir, train=True, transform=test_transform, download=True
        )
        val_set = Subset(val_set, list(range(40000, 50000)))
        train_set = Subset(train_set, list(range(40000)))
        if balanced:
            train_set, val_set = get_balanced_subset(
                train_set, number_of_samples
            )
        else:
            train_set, val_set = get_random_subset(train_set, number_of_samples)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def caltech256_dataloaders(
    batch_size=64,
    data_dir="datasets/caltech256",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
    balanced=False
):
    raise ValueError("caltech256 is not supported yet")

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        train_set = Subset(
            Caltech256(data_dir, train=True, transform=train_transform, download=True),
            list(range(int(24486 * subset_ratio))),
        )
        val_set = Subset(
            Caltech256(data_dir, train=True, transform=test_transform, download=True),
            list(range(24486, 30607)),
        )

    elif number_of_samples is not None:
        train_set = Caltech256(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = Caltech256(
            data_dir, train=True, transform=test_transform, download=True
        )
        import pdb;pdb.set_trace()
        train_set, val_set = get_balanced_subset(
            train_set, val_set, number_of_samples, val_ratio=0.2
        )

    test_set = Caltech256(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader



def caltech101_dataloaders(
    batch_size=64,
    data_dir="datasets/caltech101",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
    balanced = False
):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    Gray2RGB = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Gray2RGB,
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), Gray2RGB, normalize])

    if subset_ratio is not None:
        train_set = Subset(
            Caltech101(data_dir, transform=train_transform, download=True),
            list(range(int(7317  * subset_ratio))),
        )
        val_set = Subset(
            Caltech101(data_dir, transform=test_transform, download=True),
            list(range(7317, 9146)),
        )

    elif number_of_samples is not None:
        train_set = Caltech101(
            data_dir, transform=train_transform, download=True
        )

        val_set = Caltech101(
            data_dir, transform=train_transform, download=True
        )

        random.seed(1234)
        idxs = np.arange(len(train_set))
        random.shuffle(idxs)
        train_idxs = random.sample(list(idxs[:6117]), number_of_samples)
        val_idxs = idxs[6117:]
        train_set = Subset(train_set, list(train_idxs))
        val_set = Subset(val_set, list(val_idxs))
        # train_set, val_set = get_balanced_subset(
        #     train_set, val_set, number_of_samples, val_ratio=0.2
        # )

    # test_set = Caltech101(data_dir, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    # import pdb;pdb.set_trace()
    return train_loader, val_loader, test_loader

def svhn_dataloaders(
    batch_size=64,
    data_dir="datasets/svhn",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
    balanced=False,
):
    raise ValueError("svhn code needs to be updated")
    normalize = transforms.Normalize(
        mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
    )
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if subset_ratio is not None:
        raise ValueError("subset ratio is not supported for svhn")

    elif number_of_samples is not None:
        train_set = SVHN(
            data_dir, split="train", transform=train_transform, download=True
        )
        val_set = SVHN(data_dir, split="train", transform=test_transform, download=True)
        if balanced:
            train_set, val_set = get_balanced_subset(
                train_set, val_set, number_of_samples, val_ratio=0.2
            )
        else:
            train_set = get_random_subset(train_set, number_of_samples, val_ratio=0.2)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = Subset(
        SVHN(data_dir, split="train", transform=train_transform, download=True),
        list(range(68257)),
    )
    val_set = Subset(
        SVHN(data_dir, split="train", transform=train_transform, download=True),
        list(range(68257, 73257)),
    )
    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def fashionmnist_dataloaders(batch_size=64, data_dir="datasets/fashionmnist"):

    normalize = transforms.Normalize(mean=[0.1436], std=[0.1609])
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = Subset(
        FashionMNIST(data_dir, train=True, transform=train_transform, download=True),
        list(range(55000)),
    )
    val_set = Subset(
        FashionMNIST(data_dir, train=True, transform=test_transform, download=True),
        list(range(55000, 60000)),
    )
    test_set = FashionMNIST(
        data_dir, train=False, transform=test_transform, download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
