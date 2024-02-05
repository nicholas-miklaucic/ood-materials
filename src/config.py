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

    # Layers to use inside SAL for gradients.
    grad_layers: list[str] = field(
        default=[
            'fc16.weight',
        ],
        is_mutable=True,
    )

    def show_grad_layers(self, model):
        """Shows memory usage for grad layers."""
        param_names = dict(model.named_parameters())
        table = []
        for layer in self.grad_layers:
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
    lr_weight_epsilon: float = 1e-5

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
    # Batch size.
    batch_size: int = 128

    # Loaders to use. Always includes the R splits.
    log_loaders: tuple[str] = ('piezo',)

    # Use a very small dataset. Only use for testing.
    use_test_mode: bool = False

    # If training using data split from a test set, controls the seed.
    test_split_seed: int = 3141


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

    # Early stopping.
    early_stop: int = 500

    # Max epochs.
    max_epochs: int = 1000

    # Network learning rate.
    network_lr: float = 1e-2

    # Network loss function.
    network_loss: LossFn = LossFn.mse

    # Whether to use the new SAL code.
    use_new_sal: bool = True

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
            self.data.use_test_mode = True
            self.log.log_adv_data = False

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
