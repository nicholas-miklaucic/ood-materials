import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Optional
import typing
import pandas as pd

import pyrallis
import torch
from pyrallis import field

from utils import sizeof_fmt

pyrallis.set_config_type('toml')


class LossFn(Enum):
    mse = 'mse'
    smooth_l1 = 'smooth_l1'
    huber = 'huber'
    l1 = 'l1'


class NormFn(Enum):
    batch = 'BatchNorm1d'
    layer = 'LayerNorm'


# self.fc128 = nn.Linear(128, 128)
# self.fc64 = nn.Linear(64, 64)
# self.fc16 = nn.Linear(16, 16)

# self.bn128 = nn.BatchNorm1d(128)
# self.bn64 = nn.BatchNorm1d(64)
# self.bn16 = nn.BatchNorm1d(16)

# self.relu = nn.ReLU()
# self.inputlayer = nn.Linear(input_size, 128)

# self.con128_64 = nn.Linear(128, 64)
# self.con64_16 = nn.Linear(64, 16)
# self.output16 = nn.Linear(16, 1)


@dataclass
class ModelConfig:
    """Configuration for model."""

    # The dimensions of the layers of the network.
    layer_dims: list[int] = field(default=[2048, 1024, 512], is_mutable=True)

    # Dropout amount to use. 0 disables dropout. 1 disables every weight.
    dropout_prop: float = 0.5

    # Normalization layer to use.
    norm_type: NormFn = NormFn.layer

    # Layers to use inside SAL for gradients. 'all' means all trainable parameters.
    grad_layers: list[str] = field(
        default=[
            'all',
        ],
        is_mutable=True,
    )

    # Number of epochs to wait before starting EMA averaging.
    ema_start_epoch: int = 20

    # EMA decay parameter.
    ema_decay: float = 0.999

    def get_grad_layers(self, model):
        """Gets grad layers to use, interpreting 'all' to mean every layer."""
        if 'all' in self.grad_layers:
            return [name for (name, vals) in model.named_parameters()]
        else:
            return self.grad_layers

    def show_grad_layers(self, model):
        """Shows memory usage for grad layers."""
        param_names = dict(model.named_parameters())
        table = []
        for layer in self.get_grad_layers(model):
            if layer not in param_names:
                raise ValueError(
                    f'Layer {layer} is not in model!\nModel params:\n'
                    + pprint(list(param_names.keys()))
                )

            val = model.get_parameter(layer)
            table.append([layer, list(val.shape), sizeof_fmt(val.element_size() * val.numel())])

        table = pd.DataFrame(table, columns=['Layer', 'Shape', 'Size'])
        return table.to_string(index=None)


@dataclass
class SALConfig:
    """Configuration for SAL optimizer."""

    # The loss function used for comparing different y values.
    target_loss: LossFn = LossFn.mse

    # The weight given to ensuring adversarial inputs are close, compared to ensuring that they give
    # different predictions. Higher values create farther away adversarial examples in X-space.
    # gamma: float = 0.01

    # Factor applied to gradients of model parameters when combined with the model's layer loss.
    xa_grad_reduce: float = 0.01

    # Very suspicious float.
    deltaall: float = 20

    # Controls shrinkage of the extrema towards the mean. Extremely suspicious...
    alpha: float = 0.5

    # Attacker learning rate.
    attack_lr: float = 7e-2

    # Attacker epochs.
    attack_epochs: int = 2

    # SAL parameter learning rate.
    theta_lr: float = 1e-2

    # SAL parameter epoch number.
    theta_epochs: int = 10

    # Number of epochs between restarting adversarial data generation.
    adv_reset_epochs: int = 5

    # Epsilon used to ensure numerical stability of weights. Setting to 0 matches the original SAL
    # code, but can cause crashes due to numerical instability.
    lr_weight_epsilon: float = 1e-12

    # Number of environments to use in R(θ(w)) calculation. Must be bigger than 1 to make
    # mathematical sense. Values are randomly split from adv_data.
    r_theta_splits: int = 5

    @property
    def loss_criterion(self):
        """Gets the target loss function. Default: mse_loss."""
        return getattr(torch.nn.functional, f'{self.target_loss.value}_loss')


@dataclass
class TargetConfig:
    # Name of column used for targets.
    col_name: str = 'delta_e'


@dataclass
class DataConfig:
    # Target column. Should be one of magmom_pa, bandgap, or delta_e.
    target: str = 'delta_e'

    # Batch size.
    batch_size: int = 256

    # If training using data split from a test set, controls the seed.
    test_split_seed: int = 3141

    # Dataset split to use. 7 is the least data, 0 is everything. The relative counts:
    # 84190, 41437, 20394, 10037, 4939, 2430, 1196, 588.
    dataset_split: int = 0

    # Datasets to use for logging.
    test_sets: list[str] = field(
        default=[
            'Xshift_tsne',
            'Xshift_umap',
            'statY_delta_e',
            'infoY_delta_e',
            'statY_bandgap',
            'infoY_bandgap',
            'Rsplt1',
            'Rsplt2',
            'Rsplt3',
            'Rsplt4',
            'Rsplt5',
            'piezo',
        ],
        is_mutable=True,
    )

    # Test sets to keep unsplit, ignoring dataset_split. Use for small sets to avoid splitting them
    # down to nothing.
    no_split_test_sets: list[str] = field(default=['piezo'], is_mutable=True)


@dataclass
class PartialTrainConfig:
    # Number of epochs between partial training set regeneration.
    epochs_between_regens: int = 5

    # Partial batches.
    partial_batch: int = 5


@dataclass
class LogConfig:
    # How many epochs to wait in between partial training set regenerations.
    # Floored to be divisible by epochs_between_regens.
    partial_regens_per_log: int = 10

    # Where logging data goes. Relative to current directory.
    exp_base_dir: Path = Path('exps')

    # Whether to log adversarial data.
    log_adv_data: bool = True

    # Name of experiment. If None, use time.
    exp_name: typing.Optional[str] = None


class LoggingLevel(Enum):
    """The logging level."""

    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


@dataclass
class CLIConfig:
    # Verbosity of output.
    verbosity: LoggingLevel = LoggingLevel.info
    # Whether to show progress bars.
    show_progress: bool = True

    # Whether to log CUDA memory to debug output. Only relevant if visibility is set to debug.
    show_cuda_memory: bool = False

    def set_up_logging(self):
        from rich.logging import RichHandler

        logging.basicConfig(
            level=self.verbosity.value,
            format='%(message)s',
            datefmt='[%X]',
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    show_time=False,
                    show_level=False,
                    show_path=False,
                )
            ],
        )


@dataclass
class MainConfig:
    # Removes almost all computation by limiting data size and epochs. Use for
    # testing code.
    smoke_test: bool = False

    # Number of epochs done with the model before adversarial training begins.
    pre_adv_epochs: int = 4

    # Dataset to use for adversarial training. If None, use training data.
    adv_train_data: typing.Optional[str] = None

    # Early stopping.
    early_stop: int = 500

    # Max epochs.
    max_epochs: int = 1000

    # Network learning rate.
    network_lr: float = 1e-2

    # Network loss function.
    network_loss: LossFn = LossFn.mse

    # Random seed for reproducibility. If null, then don't seed the RNG.
    rng_seed: Optional[int] = 2718

    partial: PartialTrainConfig = field(default_factory=PartialTrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    sal: SALConfig = field(default_factory=SALConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if self.smoke_test:
            self.max_epochs = 6
            self.pre_adv_epochs = 1
            self.partial.epochs_between_regens = 2
            self.data.batch_size = 10
            self.data.dataset_split = 7
            self.model.ema_start_epoch = 3

    def seed_torch_rng(self):
        import torch

        if self.rng_seed is not None:
            torch.manual_seed(self.rng_seed)

    @property
    def partial_epoch_save(self) -> int:
        """Partial train data saved every this many epochs."""
        return self.log.partial_regens_per_log * self.partial.epochs_between_regens

    @property
    def loss_criterion(self):
        """Gets the network's loss function. Default: mse_loss."""
        return getattr(torch.nn.functional, f'{self.network_loss.value}_loss')


if __name__ == '__main__':
    from pathlib import Path

    from rich.prompt import Confirm

    if Confirm.ask('Generate configs/defaults.toml and configs/minimal.toml?'):
        default_path = Path('configs') / 'defaults.toml'
        minimal_path = Path('configs') / 'minimal.toml'

        default = MainConfig()

        with open(default_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile)

        with open(minimal_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile, omit_defaults=True)

        with default_path.open('r') as conf:
            pyrallis.cfgparsing.load(MainConfig, conf)
