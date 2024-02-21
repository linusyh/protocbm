import torch


def get_activation_fn(name="relu"):
    if name == "relu":
        return torch.nn.ReLU
    elif name == "leakyrelu":
        return torch.nn.LeakyReLU
    elif name == "sigmoid":
        return torch.nn.Sigmoid
    elif name == "selu":
        return torch.nn.SELU
    elif name == "tanh":
        return torch.nn.Tanh
    elif name in ["none", "identity", None]:
        return torch.nn.Identity
    else:
        raise ValueError(name)


def get_optimiser(name: str):
    lookup = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
    }
    return lookup.get(name.lower(), torch.optim.Adam)