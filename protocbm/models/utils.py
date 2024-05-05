from typing import Optional
import torch
import torchvision


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


def get_backbone(arch: str, 
                 output_dim: Optional[int], 
                 pretrained: bool = True):
    """
    Retrieves the backbone model for a given architecture.

    Args:
        arch (str): The architecture name. Supported architectures: 
            - "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
            - "efficientnet-v2-s", "efficientnet-v2-m", "efficientnet-v2-l"
        output_dim (int or None): The number of output dimensions. If None, the final model layer will be replaced by identity
        pretrained (bool, optional): Whether to load pretrained weights for the backbone. Defaults to True.

    Returns:
        torch.nn.Module: The backbone model.

    Raises:
        ValueError: If the specified architecture is not supported.

    Examples:
        >>> backbone = get_backbone("resnet50", output_dim=1000, pretrained=True)
    """
    
    arch = arch.lower().strip()
    if arch.startswith("resnet"):
        if arch == "resnet18":
            backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet34":
            backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet50":
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet101":
            backbone = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet152":
            backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT if pretrained else None)
        if output_dim is not None:
            backbone.fc = torch.nn.Linear(backbone.fc.in_features, output_dim)        
    elif arch.startswith("efficientnet-v2"):
        if arch == "efficientnet-v2-s":
            backbone = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        elif arch == "efficientnet-v2-m":
            backbone = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None)
        elif arch == "efficientnet-v2-l":
            backbone = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None)
        if output_dim is not None:
            backbone.classifier[1] = torch.nn.Linear(backbone.classifier[1].in_features, output_dim)
    elif arch == "synth_extractor":
        output_dim = output_dim or 128
        return torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, output_dim),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return backbone


def aggregate_predictions(predictions):
    assert len(predictions) > 0, "Predictions list is empty."
    
    output = {key: [] for key in predictions[0].keys()}
    for pred in predictions:
        for key, value in pred.items():
            output[key].append(value)
    return {
        k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v for k, v in output.items()
    }