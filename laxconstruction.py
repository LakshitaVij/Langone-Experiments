"""
This module provides methods to select and generate the correct PyTorch Lightning 
ResNet3D implementation based on config settings.
"""
import torch

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.branched_3Dresnet import DualSeriesModel
from src.models.ResNet3D.branched_3Dresnet import QuadSeriesModel
from src.models.ResNet3D.lakshita_earlyfusion import TriSeriesModel
from src.models.ResNet3D.caf import CrossAttentionFusionModel


def load_med3d_weights(model, checkpoint):
    """Go through the model state_dict and load the Med3d weights.

    The Med3D weights are from "Med3D: Transfer Learning for 3D Medical Image
    Analysis" by Chen et al. (2019) and can be found here:
    https://github.com/Tencent/MedicalNet

    Args:
        model (pl.LightningModule): The model to load the weights into.
        checkpoint (dict): The checkpoint containing the weights.
    Returns:
        model (pl.LightningModule): The model with the weights loaded.
    """

    for param_tensor in model.state_dict():
        # because of the branched architecture, the med3d state_dict key names are
        # different, we need to initialize weights for each branch with the
        # corresponding weights from the checkpoint
        new_param_tensor = "module." + param_tensor.replace(
            "resnet_branch1.", ""
        ).replace("resnet_branch2.", "").replace("resnet_branch3.", "")
        if new_param_tensor in checkpoint["state_dict"]:
            if (
                model.state_dict()[param_tensor].size()
                == checkpoint["state_dict"][new_param_tensor].size()
            ):
                # according to https://discuss.pytorch.org/t/loading-a-specific-layer-from-checkpoint/52725/2
                # this is the correct way to load a specific layer
                with torch.no_grad():
                    model.state_dict()[param_tensor].copy_(
                        checkpoint["state_dict"][new_param_tensor]
                    )
    return model


def load_saved_resnet3d_weights(model, checkpoint, exclude_layers=None, disable_gradient=False):
    """Go through the model state_dict and load the checkpoint weights.

    Args:
        model (pl.LightningModule): The model to load the weights into.
        checkpoint (dict): The checkpoint containing the weights.
        exclude_layers (list): list containing layers to exclude from loading
        disable_gradient (list): load model with/without gradient calculation enabled
    Returns:
        model (pl.LightningModule): The model with the weights loaded.
    """
    if exclude_layers is None:
        exclude_layers = []

    for param_tensor in model.state_dict():
        if param_tensor in checkpoint["state_dict"]:
            if not any(param_tensor.startswith(layer) for layer in exclude_layers):
                if (
                    model.state_dict()[param_tensor].size()
                    == checkpoint["state_dict"][param_tensor].size()
                ):
                    # according to https://discuss.pytorch.org/t/loading-a-specific-layer-from-checkpoint/52725/2
                    # this is the correct way to load a specific layer
                    if disable_gradient:
                        with torch.no_grad():
                            model.state_dict()[param_tensor].copy_(
                                checkpoint["state_dict"][param_tensor]
                            )
                    else:
                        model.state_dict()[param_tensor].copy_(
                            checkpoint["state_dict"][param_tensor]
                        )

    return model


def generate_resnet3d(config):
    """
    Generate a ResNet3D model.

    Args:
        config (dict): Dictionary containing the configuration.
    return:
        model (pl.LightningModule)
    """

    if len(config["data"]["series"]) == 4:
        model = QuadSeriesModel(config)
    elif len(config["data"]["series"]) == 3:
        fusion_type = config.get("fusion_type", "early")
        if fusion_type == "crossattn":
            model = CrossAttentionFusionModel(config)
        else:
            model = TriSeriesModel(config)
    elif len(config["data"]["series"]) == 2:
        model = DualSeriesModel(config)
    elif len(config["data"]["series"]) == 1:
        model = Base3DResNet(config)
    else:
        raise ValueError(
            f"Unsupported number of series ({len(config['data']['series'])})."
        )

    # load pretrained weights if specified
    init_weights = config["model_weights"]["load_weights"]
    if init_weights:
        print("loading pretrained weights...")

        model_ckpt = config["model_weights"].get("model_ckpt")
        t2_ckpt = config["model_weights"].get("t2_model_ckpt")
        dwi_ckpt = config["model_weights"].get("dwi_model_ckpt")
        disable_gradient = config["model_weights"]["disable_gradient"]

        if model_ckpt:
            checkpoint = torch.load(
                model_ckpt,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            model = load_saved_resnet3d_weights(
                model,
                checkpoint,
                exclude_layers=["fc", "tabular_encoder", "query_proj", "key_proj", "value_proj"],
                disable_gradient=disable_gradient
            )
        elif model.__class__.__name__ == "TriSeriesModel":
            t2_checkpoint = torch.load(
                t2_ckpt,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            model.resnet_single_branch = load_saved_resnet3d_weights(
                model.resnet_single_branch,
                t2_checkpoint,
                exclude_layers=["fc"],
                disable_gradient=disable_gradient
            )

            if model.stack_adc_b1500:
                dwi_checkpoint = torch.load(
                    dwi_ckpt,
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                model.resnet_dual_branch1 = load_saved_resnet3d_weights(
                    model.resnet_dual_branch1,
                    dwi_checkpoint,
                    exclude_layers=["fc"],
                    disable_gradient=disable_gradient
                )
            else:
                raise NotImplementedError
        else:
            print("Pretrained weights are not supported for this model type.")

    else:
        print("training from scratch...")

    return model