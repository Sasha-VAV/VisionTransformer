import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader


def load_data(
    path_to_train_data: str | None = None,
    path_to_test_data: str | None = None,
    image_size: int = 224,
    train_batch_size: int = 4,
    test_batch_size: int = 4,
    is_augmentation: bool = False,
    is_shuffle_train: bool = True,
    num_workers: int = 0,
    is_use_build_in_train: bool = False,
    is_use_build_in_test: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Function to load the training and testing data
    :param path_to_train_data: string path to train data
    :param path_to_test_data: string path to test data
    :param image_size: size of image
    :param train_batch_size: number of samples per train batch
    :param test_batch_size: number of samples per test batch
    :param is_augmentation: whether to use data augmentation or not
    :param is_shuffle_train: whether to shuffle train data or not
    :param num_workers: number of workers for data loading
    :param is_use_build_in_train: set True if you want to use torch datasets
    :param is_use_build_in_test: set True if you want to use torch datasets
    :return: tuple[DataLoader, DataLoader]
    """
    augmented_data_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(179),
            transforms.RandomVerticalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    default_data_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if is_use_build_in_train:
        # Change build in dataset here:
        from torchvision.datasets import FGVCAircraft as TrainDataset
    else:
        from torchvision.datasets import ImageFolder as TrainDataset
    if is_use_build_in_test:
        # Change build in dataset here:
        from torchvision.datasets import FGVCAircraft as TestDataset
    else:
        from torchvision.datasets import ImageFolder as TestDataset
    if path_to_train_data is not None:
        try:
            if is_augmentation:
                if not is_use_build_in_train:
                    train_data = TrainDataset(
                        root=path_to_train_data, transform=augmented_data_transform
                    )
                else:
                    # Replace this string if you want to use build in datasets
                    train_data = TrainDataset(
                        root="data/",
                        split="train",
                        transform=augmented_data_transform,
                        download=True,
                    )
                train_data_loader = DataLoader(
                    train_data,
                    batch_size=train_batch_size,
                    shuffle=is_shuffle_train,
                    num_workers=num_workers,
                    drop_last=True,
                )
            else:
                if not is_use_build_in_test:
                    train_data = TrainDataset(
                        root=path_to_train_data, transform=default_data_transform
                    )
                else:
                    train_data = TrainDataset(
                        root="data/",
                        split="train",
                        transform=default_data_transform,
                        download=True,
                    )

                train_data_loader = DataLoader(
                    train_data,
                    batch_size=train_batch_size,
                    shuffle=is_shuffle_train,
                    num_workers=num_workers,
                    drop_last=True,
                )
        except FileNotFoundError:
            train_data_loader = None
    else:
        train_data_loader = None
    if path_to_test_data is not None:
        try:
            if not is_use_build_in_test:
                test_data = TestDataset(
                    root=path_to_test_data, transform=default_data_transform
                )
            else:
                test_data = TestDataset(
                    root="data/",
                    split="test",
                    transform=default_data_transform,
                    download=True,
                )
            test_data_loader = DataLoader(
                test_data,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
            )
        except FileNotFoundError:
            test_data_loader = None
    else:
        test_data_loader = None
    return train_data_loader, test_data_loader


def load_image(path: str, img_size: int = 224) -> torch.Tensor:
    """
    Method to convert image to tensor
    :param path: path to image
    :param img_size: size, that image should be resized for
    :return: tensor that represents image
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),  # Optional: Resize the image
            transforms.CenterCrop(img_size),  # Optional: Center crop the image
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load the image
    img = Image.open(path)

    # Apply the transform to the image
    img_tensor = transform(img)

    # Add a batch dimension (since the model expects a batch of images)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor
