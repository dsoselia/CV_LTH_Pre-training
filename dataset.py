from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    SVHN,
    FashionMNIST,
    Caltech256,
    Caltech101,
)
from torch.utils.data import DataLoader, Subset,Dataset
import numpy as np
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


def get_balanced_subset(dataset, val_dataset, number_of_samples, val_ratio=0.2):
    number_of_validation_samples = int(number_of_samples / (1 - val_ratio) * val_ratio)
    if number_of_validation_samples + number_of_samples > len(dataset):
        raise ValueError("number of samples is too large")
    unique_labels = np.unique(dataset.targets)
    train_idxs = []
    for label in unique_labels:
        number_of_samples_per_label = int(number_of_samples / len(unique_labels))

        idxs = np.where(np.array(dataset.targets) == label)[0].tolist()
        train_idxs += idxs[:number_of_samples_per_label]

    dataset_train = FewShotSubset(dataset, train_idxs)
    return dataset_train, val_dataset


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
        val_set = Subset(val_set, list(range(4000, 50000)))
        train_set = Subset(train_set, list(range(4000)))
        if balanced:
            train_set, val_set = get_balanced_subset(
                train_set, val_set, number_of_samples, val_ratio=0.2
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
                train_set, val_set, number_of_samples, val_ratio=val_ratio
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
    balanced=False,
):
    data, labels = caltech(data_dir)
    (X, x_val, Y, y_val) = train_test_split(
        data, labels, test_size=0.15, stratify=labels, random_state=42
    )
    (x_train, x_test, y_train, y_test) = train_test_split(
        X, Y, test_size=0.15, random_state=42
    )
    print(
        f"x_train examples: {x_train.shape}\nx_test examples: {x_test.shape}\nx_val examples: {x_val.shape}"
    )

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [   
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        transforms.ToPILImage(),transforms.Resize((224, 224)),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(224, padding=16), transforms.ToTensor(), normalize
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = CaltechDataset(x_train, y_train, train_transform)
    val_set = CaltechDataset(x_val, y_val, val_transform)
    test_set = CaltechDataset(x_test, y_test, val_transform)

    if subset_ratio is not None:
        raise ValueError("subset ratio is not supported for caltech256")

    elif number_of_samples is not None:
        if balanced:
            train_set, val_set = get_balanced_subset(
                train_set, val_set, number_of_samples, val_ratio=val_ratio
            )
        else:
            train_set = get_random_subset(train_set, number_of_samples)

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
    balanced=False,
):

    data, labels = caltech(data_dir)
    (X, x_val, Y, y_val) = train_test_split(
        data, labels, test_size=0.15, stratify=labels, random_state=42
    )
    (x_train, x_test, y_train, y_test) = train_test_split(
        X, Y, test_size=0.15, random_state=42
    )
    print(
        f"x_train examples: {x_train.shape}\nx_test examples: {x_test.shape}\nx_val examples: {x_val.shape}"
    )

    
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToPILImage(),transforms.Resize((256, 256)),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(224, padding=16), transforms.ToTensor(), normalize]
    )

    train_set = CaltechDataset(x_train, y_train, train_transform)
    val_set = CaltechDataset(x_val, y_val, test_transform)
    test_set = CaltechDataset(x_test, y_test, test_transform)

    if subset_ratio is not None:
        raise ValueError("subset ratio is not supported for caltech101")

    elif number_of_samples is not None:
        if balanced:
            train_data, val_data = get_balanced_subset(
                train_set, val_set, number_of_samples, val_ratio=val_ratio
            )
        else:
            train_set = get_random_subset(train_set, number_of_samples)

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


def caltech(dir):
    image_paths = list(paths.list_images(dir))

    data = []
    labels = []
    for img_path in image_paths:
        label = img_path.split(os.path.sep)[-2]
        if label == "BACKGROUND_Google":
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data.append(img)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


class CaltechDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.labels = labels
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.images[index][:]

        if self.transforms:
            data = self.transforms(data)

        if self.labels is not None:
            return (data, self.labels[index])
        else:
            return data
