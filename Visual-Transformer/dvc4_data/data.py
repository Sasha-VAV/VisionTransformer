import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, FGVCAircraft
from dogs_vs_cats4.dvc4_config import (
    is_use_build_in_train,
    is_use_build_in_test,
    img_size,
)


def load_data(
    path_to_train_data: str | None = None,
    path_to_test_data: str | None = None,
    image_size: int = 224,
    train_batch_size: int = 4,
    test_batch_size: int = 4,
    is_augmentation: bool = False,
    is_shuffle_train: bool = True,
    num_workers: int = 0,
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
    if path_to_train_data is not None:
        try:
            if is_augmentation:
                if not is_use_build_in_train:
                    train_data = ImageFolder(
                        root=path_to_train_data, transform=augmented_data_transform
                    )
                else:
                    # Replace this string if you want to use build in datasets
                    train_data = FGVCAircraft(
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
                    train_data = ImageFolder(
                        root=path_to_train_data, transform=default_data_transform
                    )
                else:
                    # Replace this string if you want to use build in datasets
                    train_data = FGVCAircraft(
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
                test_data = ImageFolder(
                    root=path_to_test_data, transform=default_data_transform
                )
            else:
                test_data = FGVCAircraft(
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


def load_image(path: str):
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
