"""
Initial file, to load all the functions
"""

import torch
import torchvision.models as models
from omegaconf import DictConfig
from torch import nn

from .utils import convert_to_abs_path
from .data import load_data, load_image
from .model import ViT
from .test import test_model
from .train import train_model


class Program:
    def __init__(self, cfg: DictConfig, abs_path: str):
        self.abs_path = abs_path

        base = cfg["base"]
        modes = base["modes"]
        self.is_train = modes["train"] if modes.__contains__("train") else False
        self.is_test = modes["test"] if modes.__contains__("test") else False
        self.is_exec = modes["exec"] if modes.__contains__("exec") else False
        model_conf = base["model"]
        self.device = torch.device(
            model_conf["device"] if torch.cuda.is_available() else "cpu"
        )
        self.img_size = model_conf["img_size"]
        self.num_classes = model_conf["number_of_classes"]
        self.batch_size = model_conf["batch_size"]
        self.path_to_nn_params = convert_to_abs_path(
            self.abs_path, model_conf["path_to_nn_params"]
        )
        self.is_use_torch_vit = model_conf["is_use_torch_vit"]
        if self.is_use_torch_vit:
            torch_vit = model_conf["torch_vit"]
            self.path_to_pretrained_params = torch_vit["path_to_pretrained_params"]
            if self.path_to_pretrained_params == "None":
                self.path_to_pretrained_params = None
            else:
                self.path_to_pretrained_params = convert_to_abs_path(
                    self.abs_path, self.path_to_pretrained_params
                )
            pretrained_model = models.vit_b_16(image_size=self.img_size)
            pretrained_model.to(self.device)
            self.vit = pretrained_model
        else:
            my_vit = model_conf["vit"]
            self.vit = ViT(
                img_size=self.img_size,
                patch_size=my_vit["patch_size"],
                in_chans=my_vit["in_channels"],
                num_classes=self.num_classes,
                embed_dim=my_vit["embed_dim"],
                depth=my_vit["depth"],
                num_heads=my_vit["num_heads"],
                mlp_ratio=my_vit["mlp_ratio"],
                drop_rate=my_vit["drop_rate"],
                qkv_bias=my_vit["qkv_bias"],
            ).to(self.device)

        run = cfg["run"]

        dataset = run["dataset"]
        self.path_to_train_data = convert_to_abs_path(
            self.abs_path, dataset["path_to_train_data"]
        )
        self.path_to_test_data = convert_to_abs_path(
            self.abs_path, dataset["path_to_test_data"]
        )
        self.is_aug = dataset["is_use_augmentation"]
        self.is_shuffle = dataset["is_shuffle_train_data"]
        self.num_of_train_samples = dataset["num_of_train_samples"]
        self.num_of_test_samples = dataset["num_of_test_samples"]

        training = run["training"]
        self.epochs = training["number_of_train_epochs"]
        self.is_refresh = training["is_refresh_train_data"]

        self.images = run["run"]["images"]
        self.classes = run["run"]["classes"]

        logging = run["logging"]
        self.print_n = logging["print_n"]
        self.is_use_wandb = logging["is_use_wandb"]
        if self.is_use_wandb:
            self.wandb_config = logging["wandb"]
            self.wandb_config.update({"epochs": self.epochs})

        # All values are inited from config, now we process them
        if self.is_use_wandb:
            import wandb

            wandb.init(
                # set the wandb project where this run will be logged
                project="Visual-Transformer",
                config=self.wandb_config,
            )

        if self.is_use_torch_vit:
            if self.path_to_pretrained_params is not None:
                self.vit.load_state_dict(
                    torch.load(
                        self.path_to_pretrained_params,
                        weights_only=True,
                        map_location=self.device,
                    )
                )

            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.heads = nn.Linear(self.vit.hidden_dim, self.num_classes)
            if self.path_to_pretrained_params is not None:
                torch.save(self.vit.state_dict(), self.path_to_nn_params)
            try:
                self.vit.load_state_dict(
                    torch.load(
                        self.path_to_nn_params,
                        weights_only=True,
                        map_location=self.device,
                    )
                )
            except FileNotFoundError:
                torch.save(self.vit.state_dict(), self.path_to_nn_params)

        if self.is_train or self.is_test:
            self.train_data_loader, self.test_data_loader = load_data(
                self.path_to_train_data,
                self.path_to_test_data,
                image_size=self.img_size,
                train_batch_size=self.batch_size,
                test_batch_size=self.batch_size,
                is_augmentation=self.is_aug,
                is_shuffle_train=self.is_shuffle,
                num_workers=0,
                is_use_build_in_train=False,
                is_use_build_in_test=False,
            )

        self.vit.to(self.device)

    def run(self):
        print("New run:")
        total_params = sum(p.numel() for p in self.vit.parameters())
        print("Total number of parameters:", total_params)
        if self.is_train:
            train_model(
                vit=self.vit,
                device=self.device,
                train_data_loader=self.train_data_loader,
                path_to_nn_params=self.path_to_nn_params,
                epochs=self.epochs,
                img_size=self.img_size,
                test_data_loader=self.test_data_loader,
                is_use_wandb=self.is_use_wandb,
                is_aug=self.is_aug,
                is_shuffle_train_data=self.is_shuffle,
                refresh_train_data=self.is_refresh,
                path_to_train_data=self.path_to_train_data,
                batch_size=self.batch_size,
                save_n_times_per_epoch=self.print_n,
                max_number_of_train_samples=self.num_of_train_samples,
                max_number_of_test_samples=self.num_of_test_samples,
                num_of_workers=0,
            )
        if self.is_test:
            test_model(
                vit=self.vit,
                device=self.device,
                test_data_loader=self.test_data_loader,
                path_to_nn_params=self.path_to_nn_params,
                max_test_samples=self.num_of_test_samples,
                batch_size=self.batch_size,
            )
        if self.is_exec:
            self.vit.load_state_dict(
                torch.load(
                    self.path_to_nn_params, weights_only=True, map_location=self.device
                )
            )
            for s in self.images:
                self(s)

    def __call__(self, s: str) -> int:
        """
        Method to predict image class
        :param s: path to image
        :return: class number
        """
        img_tensor = load_image(convert_to_abs_path(self.abs_path, s), self.img_size)
        img_tensor = img_tensor.to(device=self.device)
        output = self.vit(img_tensor)
        _, predicted = torch.max(output, 1)
        try:
            print("Predicted: ", " ".join(f"{self.classes[predicted.item()]:5s}"))
            return predicted.item()
        except IndexError:
            print(f"Predicted: class with number {predicted}, which is wrong, sorry")
            return -1
