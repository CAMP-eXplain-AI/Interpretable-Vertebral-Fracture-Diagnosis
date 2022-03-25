from typing import Dict, Any, Iterable
import numbers
import yaml
import os

def _prepare_config(config) -> Dict[str, Any]:
    """
    Returns the final configuration used by the model. May be used to clean user input,
    sort items, etc.
    """

    classes_per_task = {'detection': 1, 'grading': 6, 'simple_grading': 4}
    NUM_CLASSES = classes_per_task[config["TASK"]]

    return {
        "learning_rate": config["LEARNING_RATE"],
        "model_name": config["MODEL_NAME"],
        "dataset": "VerSe",
        "batch_size": config["BATCH_SIZE"], 
        "input_size": config["INPUT_SIZE"],
        "input_dim": config["INPUT_DIM"],
        "mask": config["MASK"],
        "oversampling": config["OVERSAMPLING"],
        "fold": config["FOLD"],
        "dropout": config["DROPOUT"],
        "frozen_layers": config["FROZEN_LAYERS"],
        "num_classes": NUM_CLASSES,
        "early_stopping_patience": config["EARLY_STOPPING_PATIENCE"],
        "min_vertebrae_level": config["MIN_VERTEBRAE_LEVEL"],
        "dataset_path": config["DATASET_PATH"],
        "loss": config["LOSS"], 
        "weighted_loss": config["WEIGHTED_LOSS"],
        "transforms": sorted(config["TRANSFORMS"]),
        "task": config["TASK"],

        # passed through, will not be part of final hyperparameters
        "USE_WANDB": config["USE_WANDB"],
        "WANDB_API_KEY": config["WANDB_API_KEY"],
        "SAVE_MODEL": config["SAVE_MODEL"]
    }

def _sanity_check_config(config):
    """
    Runs simple assertions to test that the config file is actually valid.
    """
    # option validity assertions
    assert any([config['mask'] == o for o in ['none', 'channel', 'apply', 'apply_all', 'crop']])
    assert os.path.exists(config["dataset_path"])

    # datatype assertions
    for numeric_key in ["batch_size", "input_size", "input_dim", "dropout", "early_stopping_patience", "min_vertebrae_level", "fold"]:
        assert isinstance(config[numeric_key], numbers.Number)

    for list_key in ["transforms", "frozen_layers"]:
        assert isinstance(config[list_key], Iterable)

    # logic assertions
    assert not (config['task'] == 'detection' and config['loss'] != 'binary_cross_entropy' and config['loss'] != 'focal')

    # ensure models fit the data
    nets_3d = ["UNet3D"]
    for net_3d in nets_3d:
        assert not (net_3d in config['model_name']) or config['input_dim'] == 3

    if config['oversampling'] and config['weighted_loss']:
        print("Oversampling as well as weighted loss are enabled, you may want to disable one")

    if config['loss'] == 'focal' and config['weighted_loss']:
        print("Focal loss does not support manual class weighting")

    if config['loss'] == 'focal' and config['oversampling']:
        print("Focal loss and oversampling are enabled, you may want to disable oversampling")

def get_config(config_file_path: str):
    """
    Retrieves the configuration from the given file path after running a sanity check and
    pre-processing steps.
    """

    with open(config_file_path, 'r') as stream:
        config_stream = yaml.load(stream, Loader=yaml.FullLoader)
        config = _prepare_config(config_stream)
        _sanity_check_config(config)
        return config