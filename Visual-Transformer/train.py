import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import ViT
from .test import test_model
from .data import load_data


def train_model(
    vit: ViT,
    device: torch.device,
    train_data_loader: DataLoader,
    path_to_nn_params: str,
    epochs: int = 10,
    img_size: int = 224,
    test_data_loader: DataLoader | None = None,
    is_use_wandb: bool = False,
    is_aug: bool = False,
    is_shuffle_train_data: bool = False,
    refresh_train_data: bool = False,
    path_to_train_data: str | None = None,
    batch_size: int = 4,
    save_n_times_per_epoch: int = 5,
    max_number_of_train_samples: int = 24000,
    number_of_validation_samples: int = 100,
    max_number_of_test_samples: int = -1,
    num_of_workers: int = 0,
):
    """
    Function to train model
    :param vit: object of class ViT that represents model
    :param device: torch device, can be either cpu or cuda
    :param train_data_loader: object of DataLoader that represents the training data
    :param test_data_loader: object of DataLoader that represents the testing data
    :param path_to_nn_params: path to parameters of the NN
    :param epochs: number of epochs to train model
    :param img_size: size of one side of the image, that model works with
    :param is_use_wandb: whether to use wandb instead or not
    :param is_aug: whether to augmented data or not
    :param is_shuffle_train_data: whether to shuffle train data or not
    :param refresh_train_data: whether to refresh the training data or not
    :param path_to_train_data: path to training data
    :param batch_size: batch size for training
    :param save_n_times_per_epoch: number of times to save the training data
    :param max_number_of_train_samples: maximum number of train samples to load
    :param number_of_validation_samples: number of validation samples to load
    :param max_number_of_test_samples: maximum number of test samples to load
    :param num_of_workers: number of workers for data loading
    :return: nothing
    """
    if train_data_loader is None:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vit.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_data_loader) * batch_size,
        epochs=epochs,
    )

    try:
        vit.load_state_dict(torch.load(path_to_nn_params, weights_only=True))
    except IOError:
        pass

    if is_use_wandb:
        import wandb

    if max_number_of_train_samples == -1:
        max_number_of_train_samples = len(train_data_loader) * batch_size
    if max_number_of_test_samples == -1:
        max_number_of_test_samples = len(test_data_loader) * batch_size

    if save_n_times_per_epoch > 0:
        save_every_n_times_per_epoch = (
            max_number_of_train_samples // batch_size // save_n_times_per_epoch
        )
    else:
        save_every_n_times_per_epoch = max_number_of_train_samples + 1

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_loss = 0.0
        val_loss = -1
        val_correct = -1
        val_samples = 1
        number_of_images = 0
        for i, data in tqdm(enumerate(train_data_loader, 0)):
            # get the inputs;
            # data is a tuple of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            number_of_images += inputs.size(0)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = vit(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if number_of_images <= number_of_validation_samples:
                val_loss = 0.0 if val_loss == -1 else val_loss
                val_correct = 0.0 if val_correct == -1 else val_correct
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples = i + 1
            if (
                i % save_every_n_times_per_epoch == save_every_n_times_per_epoch - 1
            ):  # print every n mini-batches
                print(
                    f"[{epoch + 1}, {(i + 1) * batch_size:5d}] "
                    f"loss: {running_loss/save_every_n_times_per_epoch:.5f}"
                )
                running_loss = 0.0
                torch.save(vit.state_dict(), path_to_nn_params)

            if i * batch_size >= max_number_of_train_samples:
                break

        torch.save(vit.state_dict(), path_to_nn_params)

        accuracy = -1
        if test_data_loader is not None:
            accuracy = test_model(
                vit,
                device,
                test_data_loader,
                path_to_nn_params,
                max_test_samples=max_number_of_test_samples,
                batch_size=batch_size,
            )

        console_log, wandb_log = log_data(
            epoch=epoch + 1,
            lr=0.005,  # scheduler.get_last_lr()[0],
            train_loss=total_loss / len(train_data_loader),
            val_loss=val_loss / val_samples,
            val_acc=val_correct / (val_samples * batch_size) * 100,
            test_acc=accuracy,
        )
        print(console_log)
        if is_use_wandb:
            wandb.log(wandb_log)

        if refresh_train_data:
            train_data_loader, _ = load_data(
                path_to_train_data,
                None,
                image_size=img_size,
                train_batch_size=batch_size,
                test_batch_size=batch_size,
                is_augmentation=is_aug,
                is_shuffle_train=is_shuffle_train_data,
                num_workers=num_of_workers,
            )

    print("Finished Training")


def log_data(
    epoch: int,
    lr: float,
    train_loss: float,
    val_loss: float = -1.0,
    val_acc: float = -1.0,
    test_acc: float = -1.0,
) -> tuple:
    wandb_log = dict()
    print_log = f""

    def put_values(
        key: str, value: float | int, print_str: str, wandb_dict: dict
    ) -> tuple:
        if isinstance(value, float):
            print_str += f"{key}: {value:.5f}, "
        else:
            print_str = f"{key}: {value}, "
        wandb_dict.update({key: value})
        return print_str, wandb_dict

    print_log, wandb_log = put_values("epoch", epoch, print_log, wandb_log)
    print_log, wandb_log = put_values("lr", lr, print_log, wandb_log)
    print_log, wandb_log = put_values("train_loss", train_loss, print_log, wandb_log)
    if val_loss >= 0:
        print_log, wandb_log = put_values("val_loss", val_loss, print_log, wandb_log)
    if val_acc >= 0:
        print_log, wandb_log = put_values("val_acc", val_acc, print_log, wandb_log)
    if test_acc >= 0:
        print_log, wandb_log = put_values("test_acc", test_acc, print_log, wandb_log)

    return print_log[:-2], wandb_log
