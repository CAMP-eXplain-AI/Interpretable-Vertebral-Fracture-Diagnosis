from backbones.unet3d import UNet3D

def _freeze_layers_if_any(model, hparams):
    if len(hparams.frozen_layers) == 0:
        return model

    for (name, param) in model.named_parameters():
        if any([name.startswith(to_freeze_name) for to_freeze_name in hparams.frozen_layers]):
            param.requires_grad = False

    return model

def _replace_inplace_operations(model):
    # Grad-CAM compatibility
    for module in model.modules():
        if hasattr(module, "inplace"):
            setattr(module, "inplace", False)
    return model

def get_backbone(hparams):
    backbone = None

    in_channels = 1 + (hparams.mask == 'channel')

    if hparams.model_name == 'UNet3D':
        backbone = UNet3D(
            in_channels=in_channels,
            input_size=hparams.input_size,
            n_class=hparams.num_classes - (hparams.loss == 'ordinal_regression')
        )
    else:
          raise NotImplementedError

    backbone = _replace_inplace_operations(backbone)
    backbone = _freeze_layers_if_any(backbone, hparams)

    return backbone