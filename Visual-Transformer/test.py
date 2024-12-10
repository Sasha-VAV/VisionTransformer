import torch
from torch.utils.data import DataLoader
from model import ViT
from tqdm import tqdm


def test_model(
    vit: ViT,
    device: torch.device,
    test_data_loader: DataLoader,
    path_to_nn_params: str,
    max_test_samples: int = -1,
    batch_size: int = 32,
) -> float:
    """
    Function to dvc4_test the dvc4_model
    :param vit: object of CNN class that will be used to dvc4_test the model
    :param device: torch device can be either cpu or cuda
    :param test_data_loader: object of DataLoader that represents the test data
    :param path_to_nn_params: path to parameters of the CNN model
    :param max_test_samples: maximum number of test samples
    :return: accuracy of the model in percent
    """
    if test_data_loader is None:
        return -1

    vit.load_state_dict(torch.load(path_to_nn_params, weights_only=True))
    max_test_batches = max_test_samples // batch_size
    correct = 0
    total = 0
    for i, data in tqdm(enumerate(test_data_loader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vit(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if 0 < max_test_batches <= i:
            break

    print(f"{total} test samples. Accuracy: {correct * 100 / total}")
    return correct * 100 / total
