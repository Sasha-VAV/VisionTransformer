"""
Main class to work with CNN
"""
import torch
import wandb
from torch import nn

from dvc4_config import *
from dvc4_data import load_data, load_image
from dvc4_test import test_model
from dvc4_train import train_model


def wandb_init(is_init: bool = False):
    """
    WANDB
    :param is_init: set by default ot False if you do not want to use wandb
    :return:
    """
    if is_init:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Visual-Transformer",
            # track hyperparameters and run metadata
            config=wandb_config,
        )
    pass


def change_last_layer():
    vit.load_state_dict(torch.load(path_to_pretrained_params, weights_only=True))
    for param in vit.parameters():
        param.requires_grad = False
    vit.heads = nn.Linear(vit.hidden_dim, num_classes)
    if is_save_last_layer:
        torch.save(vit.state_dict(), path_to_nn_params)
    return vit.to(device)


if is_need_to_change_last_layer:
    vit = change_last_layer()


pytorch_total_params = sum(p.numel() for p in vit.parameters())

print(
    f"------------------------------------------\nNumber of parameters: {pytorch_total_params}\n"
    f"-------------------------------------------"
)

train_data_loader, test_data_loader = load_data(
    path_to_train_data,
    path_to_test_data,
    image_size=img_size,
    train_batch_size=train_batch_size,
    test_batch_size=train_batch_size,
    is_augmentation=is_aug,
    is_shuffle_train=is_shuffle_train_data,
    num_workers=num_of_workers,
)

if train_data_loader is not None:
    wandb_init(is_use_wandb)

train_model(
    vit=vit,
    device=device,
    train_data_loader=train_data_loader,
    path_to_nn_params=path_to_nn_params,
    epochs=num_of_train_epochs,
    test_data_loader=test_data_loader,
    is_use_wandb=is_use_wandb,
    refresh_train_data=is_refresh_train_data,
    path_to_train_data=path_to_train_data,
    batch_size=train_batch_size,
    save_n_times_per_epoch=print_n_times_per_epoch,
    max_number_of_train_samples=max_number_of_train_samples,
    number_of_validation_samples=number_of_validation_samples,
    max_number_of_test_samples=max_number_of_test_samples,
)

test_model(
    vit=vit,
    device=device,
    test_data_loader=test_data_loader,
    path_to_cnn_params=path_to_nn_params,
    max_test_samples=max_number_of_test_samples,
    batch_size=train_batch_size,
)

if list_of_images_paths is None:
    exit(0)

print("Now let's see your photos")

vit.load_state_dict(torch.load(path_to_nn_params, weights_only=True))
for s in list_of_images_paths:
    img_tensor = load_image(s)
    img_tensor = img_tensor.to(device=device)
    output = vit(img_tensor)
    _, predicted = torch.max(output, 1)
    try:
        print("Predicted: ", " ".join(f"{classes[predicted]:5s}"))
    except IndexError:
        print(f"Predicted: class with number {predicted}, which is wrong, sorry")
    # print(f"Tensor: {output}")
