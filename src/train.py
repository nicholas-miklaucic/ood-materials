# -*-coding:utf-8-*-
import logging
import math
import os
from datetime import datetime
from pathlib import Path
import pprint
from click import Parameter

import numpy as np
import pandas as pd
import pyrallis
import rich.progress as prog
import torch
import torch.nn as nn
import torch.optim as optim
from config import MainConfig, SALConfig
from einops import einsum, rearrange
from torch.autograd import grad
from torch.func import grad_and_value, jacrev
from torch.utils.data import DataLoader, Dataset, Subset
from utils import debug_shapes, debug_summarize, log_cuda_mem, same_storage, to_np

config = pyrallis.parse(MainConfig)

config.cli.set_up_logging()

config.seed_torch_rng()

torch.cuda.memory._record_memory_history()

torch.autograd.set_detect_anomaly(True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_folder_based_on_time(base_path: os.PathLike):
    base_path = Path(base_path)

    # Get the current time
    current_time = datetime.now()

    time_str = current_time.strftime('%m-%d-%H-%M')
    if config.smoke_test:
        time_str = 'smoke-test'

    # Create a folder name based on the formatted time
    folder_name = base_path / f'{config.target.col_name}_{time_str}'

    if config.log.exp_name is not None and not config.smoke_test:
        exp_name_path = base_path / f'{config.target.col_name}_{config.log.exp_name}'
        if exp_name_path.exists():
            logging.warning(
                f'Path {exp_name_path.absolute()} already exists, using {folder_name.absolute()}'
            )
        else:
            folder_name = exp_name_path

    for subfolder in ('adv', 'models', 'sorted'):
        os.makedirs(folder_name / subfolder, exist_ok=True)

    with (folder_name / 'run_params.toml').open('w') as configfile:
        pyrallis.dump(config, configfile)

    return folder_name


exp_dir = create_folder_based_on_time(Path.cwd() / config.log.exp_base_dir)


class MyDataset(Dataset):
    def __init__(self, inputs, target):
        self.inputs = inputs.values.astype(np.float32)
        self.labels = target.values.astype(np.float32)

        self.inputs = torch.from_numpy(self.inputs).to(device)
        self.labels = torch.from_numpy(self.labels).to(device)

    @property
    def dim_x(self) -> int:
        return self.inputs.shape[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index].unsqueeze(0)

    def getSALdata(self):
        input = self.inputs.cpu().numpy()
        label = self.labels.cpu().numpy()
        return (input, label)


with open('feat_col_name.txt', 'r') as file:
    column_names = file.read().split('\n')


class RecurrentDataset(Dataset):
    def __init__(self, df):
        self.inputs = df[input_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.float32)

        self.inputs = torch.from_numpy(np.stack(self.inputs)).to(device)
        self.labels = torch.from_numpy(np.stack(self.labels)).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index].unsqueeze(0)


# Step 4: Use DataLoader to create batches
batch_size = config.data.batch_size

all_data = pd.read_feather('mpc_full_feats_scaled_split.feather')
all_data.drop(columns=['comp', 'TSNE_x', 'TSNE_y', 'umap_x', 'umap_y'], inplace=True)

label_cols = ['magmom_pa', 'bandgap', 'delta_e']
all_test_sets = [
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
]

input_cols = [
    c for c in all_data.columns if c not in all_test_sets + label_cols + ['dataset_split']
]
test_set_flags = all_data[all_test_sets]

in_split = all_data['dataset_split'] >= config.data.dataset_split
inputs = all_data.loc[in_split & (~test_set_flags).all(axis=1)]

train_dataset = MyDataset(inputs[input_cols], inputs[config.data.target])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_sets = {}

for name in config.data.test_sets:
    test_set = all_data.query(name)
    if name not in config.data.no_split_test_sets:
        test_set = test_set[test_set['dataset_split'] >= config.data.dataset_split]

    dataset = MyDataset(test_set[input_cols], test_set[config.data.target])
    loader = DataLoader(dataset, batch_size=len(dataset))
    test_sets[name] = (dataset, loader)

# data = train_dataset.getSALdata()


def sample_and_remove_from_testset(test_dataloader, num_samples=config.data.batch_size):
    # Step 1: Get the entire test set
    test_dataset = test_dataloader.dataset

    # Step 2: Sample specified number of indices from the test set
    with torch.random.fork_rng() as _rng:
        torch.manual_seed(config.data.test_split_seed)
        sampled_indices = torch.randperm(len(test_dataset))

    # Select the first num_samples indices from the shuffled indices
    sampled_indices = sampled_indices[:num_samples]

    # Step 3: Create a DataLoader with the sampled test batch
    sampled_test_dataset = Subset(test_dataset, sampled_indices)
    sampled_test_dataloader = DataLoader(sampled_test_dataset, batch_size=num_samples, shuffle=True)

    # Step 4: Create a DataLoader with the remaining test set (excluding the sampled indices)
    remaining_indices = [i for i in range(len(test_dataset)) if i not in sampled_indices]
    remaining_test_dataset = Subset(test_dataset, remaining_indices)
    remaining_test_dataloader = DataLoader(
        remaining_test_dataset, batch_size=len(remaining_test_dataset), shuffle=True
    )

    # Step 5: Print the lengths of the original test set, sampled test set, and remaining test set
    logging.debug(f'Original Test Set Size: {len(test_dataset)}')
    logging.debug(f'Sampled Test Set Size: {len(sampled_test_dataloader.dataset)}')
    logging.debug(f'Remaining Test Set Size: {len(remaining_test_dataloader.dataset)}')

    return sampled_test_dataloader, remaining_test_dataloader


if config.adv_train_data is None:
    adv_loader = None
else:
    test_data, test_loader = test_sets[config.adv_train_data]
    adv_loader, remaining_loader = sample_and_remove_from_testset(test_loader)

    test_sets[config.adv_train_data] = (test_data, remaining_loader)


def save_tensor(ori_data, ori_lab, adv_data, adv_lab, epoch, batch_idx):
    import pandas as pd

    ori_data_list = ori_data.clone().cpu().numpy()
    ori_labels_list = ori_lab.clone().cpu().numpy()
    adv_data_list = adv_data.clone().cpu().numpy()
    adv_labels_list = adv_lab.clone().cpu().numpy()

    filename = exp_dir / 'adv' / f'adv_data_labels_in_attack{epoch}_{batch_idx}.feather'

    header = input_cols + ['label']

    # Write data and labels for each group
    rows_group1 = np.column_stack((ori_data_list, ori_labels_list))
    # TODO this was a typo?
    # rows_group2 = np.column_stack((adv_data_list, ori_labels_list))
    rows_group2 = np.column_stack((adv_data_list, adv_labels_list))
    all_rows = np.vstack((rows_group1, rows_group2))

    df = pd.DataFrame(all_rows, columns=header)
    df.to_feather(filename)


class IRNet_intorch(torch.nn.Module):
    #'128-64-16'
    def __init__(self, input_size):
        super(IRNet_intorch, self).__init__()
        self.fc128 = nn.Linear(128, 128)
        self.fc64 = nn.Linear(64, 64)
        self.fc16 = nn.Linear(16, 16)

        self.bn128 = nn.BatchNorm1d(128)
        self.bn64 = nn.BatchNorm1d(64)
        self.bn16 = nn.BatchNorm1d(16)

        self.relu = nn.ReLU()
        self.inputlayer = nn.Linear(input_size, 128)

        self.con128_64 = nn.Linear(128, 64)
        self.con64_16 = nn.Linear(64, 16)
        self.output16 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.inputlayer(x)

        x_res = x
        x = self.fc128(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = x + x_res
        x = self.con128_64(x)

        x_res = x
        x = self.fc64(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = x + x_res
        x = self.con64_16(x)

        x_res = x
        x = self.fc16(x)
        x = self.bn16(x)
        x = self.relu(x)
        x = x + x_res

        x = self.output16(x)
        return x


train_best_loss = float('inf')
partial_train_best_loss = float('inf')
Rsplt_ave_best_loss = float('inf')


# Optimizer Class to maximize loss of adversarial dataset
class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.9, epsilon=1e-8):
        self.device = torch.device(device)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.m_hat = None
        self.v_hat = None
        self.initialize = False

    # Grad = adv_gradient
    # Iternum = iteration
    # theta = adv_images
    # Gradient ascent
    def update(self, grad, iternum, theta):
        if not self.initialize:
            self.m = (1 - self.beta1) * grad
            self.v = (1 - self.beta2) * grad**2
            self.initialize = True
        else:
            assert self.m.shape == grad.shape
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        self.m_hat = self.m / (1 - self.beta1**iternum)
        self.v_hat = self.v / (1 - self.beta2**iternum)
        return theta + self.lr * self.m_hat / (self.epsilon + torch.sqrt(self.v_hat))


class StableAL:
    def __init__(
        self,
        model: torch.nn.Module,
        dim_x: int,
        sal_conf: SALConfig,
        grad_layers: tuple[str] = config.model.grad_layers,
    ):
        self.weights = None
        self.model = model
        self.weight_grad = None
        self.xa_grad = None
        self.theta_grad = None
        self.gamma = None
        self.adversarial_data = None
        self.loss_criterion = sal_conf.loss_criterion

        self.conf = sal_conf
        self.adv_based_on = None
        self.adv_again = None

        self.X = None
        self.y = None

        self.dim_x = dim_x
        self.grad_layers = grad_layers

        # self.model = IRNet_intorch(dim_x).to(device)
        # # Covariate Weights
        self.weights = torch.zeros(dim_x).reshape(-1, 1) + 100.0
        self.weights = self.weights.to(device)

        self.min_weight = torch.min(self.weights)
        self.attack_gamma = (1.0 / self.min_weight).data
        self.zero_list = []

    @property
    def grad_params(self) -> dict[str, Parameter]:
        """Params of the model used to take gradients."""
        return {name: self.model.get_parameter(name) for name in sorted(self.grad_layers)}

    @property
    def grad_param_size(self) -> dict[str, Parameter]:
        """Total number of parameters with SAL gradients in the model."""
        return sum([np.prod(list(x.shape)) for x in self.grad_params.values()])

    def unpack_grad_params(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Destructures values to be dict of parameter values."""
        unpacked = {}
        values = values.view(-1)
        curr_i = 0
        for name in sorted(self.grad_layers):
            value = self.model.get_parameter(name)
            unpacked[name] = values[curr_i : curr_i + value.numel()].view_as(value)
            curr_i += value.numel()

        if curr_i != len(values):
            raise ValueError(f'Unused values: {curr_i} != {len(values)}')

        return unpacked

    def pack_grad_params(self, unpacked: dict[str, torch.Tensor]) -> torch.Tensor:
        """Packs values into a single flattened tensor."""
        packed = []
        for name in sorted(unpacked):
            packed.append(unpacked[name].flatten())
        return torch.cat(packed)

    def cost_function(self, x, x_adv):
        # Variable cost level where the weights determine the cost level
        cost = torch.mean(((x - x_adv) ** 2).mm(self.weights)).to(device)
        return cost

    # Loss across Training environments
    # Self.loss_criterion = MSELoss
    def r(self, environments, alpha=None):
        if len(environments) == 1:
            # split single environment into multiple
            x, y = environments[0]
            inds = torch.randperm(len(x))
            environments = []
            for ix in torch.tensor_split(inds, self.conf.r_theta_splits):
                environments.append([x[ix], y[ix]])

        env_loss = torch.empty(
            len(environments),
            device=device,
            dtype=torch.float32,
        )
        for i, (x_e, y_e) in enumerate(environments):
            x_e = x_e.to(device)
            y_e = y_e.to(torch.float32).to(device)
            env_loss[i] = self.loss_criterion(self.model(x_e), y_e)

        # TODO consider making this softmax?
        max_index = torch.argmax(env_loss)
        min_index = torch.argmin(env_loss)

        env_loss[max_index] *= 1 + alpha
        env_loss[min_index] *= 1 - alpha
        result = torch.sum(env_loss)
        return result

    # generate adversarial data
    # Maximize the loss using their own ADAM.update method(their own optimizer)
    def attack(self, data, epoch, batch_idx):
        attack_lr = self.conf.attack_lr
        images, labels = data
        images_adv = images.clone().detach()

        optimizer = Adam(learning_rate=attack_lr)

        for i in range(self.conf.attack_epochs):
            if images_adv.grad is not None:
                images_adv.grad.data.zero_()

            images_adv = images_adv.to(device)
            images_adv.requires_grad_(True)
            outputs = self.model(images_adv)

            labels = labels.float().to(device)
            images = images.to(device)
            loss = self.loss_criterion(outputs, labels) - self.attack_gamma * self.cost_function(
                images, images_adv
            )

            loss.backward()
            images_adv.data = optimizer.update(images_adv.grad, i + 1, images_adv)

        self.weight_grad = -2 * self.attack_gamma * attack_lr * (images_adv - images)
        temp_image = images_adv.clone().detach()
        temp_label = labels.clone().detach()
        self.adversarial_data = (temp_image, temp_label)

        # save adv and ori data
        if config.log.log_adv_data:
            save_tensor(images, labels, temp_image, temp_label, epoch, batch_idx)

        return images_adv, labels

    def train_theta(
        self,
        data,
        end_flag=False,
        epoch=0,
        batch_idx=None,
    ):
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=self.conf.theta_lr)
        self.adv_based_on = data

        self.xa_grad = None
        # For __ Theta self.conf.theta_epochs
        for i_theta in range(self.conf.theta_epochs):
            if i_theta % self.conf.adv_reset_epochs == 0 or not end_flag:
                images_adv, labels = self.attack(data, epoch=epoch, batch_idx=batch_idx)

            else:
                # TODO what is this?
                self.adv_again = self.adversarial_data
                images_adv, labels = self.attack(
                    self.adversarial_data,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )

            # TODO deal with batch norm properly
            from torch.func import replace_all_batch_norm_modules_

            replace_all_batch_norm_modules_(self.model)

            optimizer.zero_grad()
            images_adv = images_adv.to(device)
            outputs = model(images_adv)
            loss = self.loss_criterion(outputs, labels.float())

            # debug_shapes("images_adv", "labels", **nuisance_params)

            def base_loss(grad_params, inputs, targets, nuisance_params):
                for name, value in self.unpack_grad_params(grad_params).items():
                    nuisance_params[name] = value
                outputs = torch.func.functional_call(model, nuisance_params, inputs)
                loss = self.loss_criterion(outputs, targets)
                return loss

            dl_th = grad_and_value(base_loss)

            # l_th_out = dl_th(grad_param, images_adv, labels, nuisance_params)
            # debug_shapes("l_th_out")

            d2l_th_x = jacrev(dl_th, argnums=1, has_aux=True)

            params = dict(self.model.named_parameters())

            # (dloss_dtheta, _loss2) = dl_th(self.grad_params, images_adv, labels, params)
            # dloss_dtheta = dloss_dtheta.reshape(-1)

            with torch.no_grad():
                (dtheta_dx, _loss3) = d2l_th_x(
                    self.pack_grad_params(self.grad_params), images_adv, labels, params
                )

            # debug_summarize(**dict(self.model.named_parameters()))

            # dtheta_dx = reduce(dtheta_dx, 'l1 l2 batch dim_x -> l1 l2 dim_x', 'sum')
            dtheta_dx = rearrange(dtheta_dx, 'theta batch dim_x -> batch theta dim_x')

            if self.xa_grad is None:
                self.xa_grad = 0
            self.xa_grad += dtheta_dx

            loss.backward(retain_graph=True)
            optimizer.step()

        if config.cli.show_cuda_memory:
            logging.debug('Memory after train_theta():')
            log_cuda_mem()

    def adv_step_new(self, data, target, end_flag, epoch, batch_idx):
        # TODO split this from adv_target and adv_data
        self.train_theta(
            (data, target),
            end_flag,
            epoch,
            batch_idx,
        )

        rtheta = self.r([[data, target]], alpha=self.conf.alpha / math.sqrt(epoch + 1))

        # debug_summarize(grad_params=self.grad_params)
        self.theta_grad = grad(
            rtheta,
            [self.model.get_parameter(n) for n in sorted(self.grad_layers)],
        )

        self.theta_grad = self.pack_grad_params(
            {n: gradient for gradient, n in zip(self.theta_grad, sorted(self.grad_layers))}
        )

        # debug_shapes(th_gr=self.theta_grad, xa=self.xa_grad, wg=self.weight_grad)

        # dR/dθ: theta
        # dθ/dX: batch theta dim_x
        # dX/dw: batch dim_x

        # If this ever becomes an issue with memory (dθ/dX doesn't fit), it should be possible to
        # do the VJP instead.
        deltaw1 = einsum(
            self.theta_grad,
            self.xa_grad * -self.conf.xa_grad_reduce,
            self.weight_grad,
            'theta, batch theta dim_x, batch dim_x -> dim_x',
        )

        if epoch == 17:
            debug_summarize(
                True,
                rt=rtheta,
                dw=deltaw1,
                tg=self.theta_grad,
                xa=self.xa_grad,
                wg=self.weight_grad,
            )

        deltaw = deltaw1
        dw_nonzero = [i for i in range(len(deltaw)) if i not in self.zero_list]

        deltaw[self.zero_list] = 0.0
        max_grad = torch.max(torch.abs(deltaw))
        deltastep = self.conf.deltaall
        lr_weight = (deltastep / (max_grad + self.conf.lr_weight_epsilon)).detach()
        logging.debug(f'RLoss: {rtheta.data:.4f}')

        self.weights -= lr_weight * deltaw.detach().reshape(self.weights.shape)

        if torch.isnan(self.weights).any():
            debug_summarize(
                show_stat=True,
                tg=self.theta_grad,
                xa=self.xa_grad,
                wg=self.weight_grad,
                dw=deltaw,
                mg=max_grad,
                lrw=lr_weight,
            )

        if epoch == 17:
            logging.debug(f'Zeroed values: {len(set(self.zero_list))}/{len(deltaw)}')
            logging.debug('Nonzero Δw {}'.format(deltaw[dw_nonzero]))
            logging.debug(f'weights_max: {torch.max(torch.abs(deltaw)).item()}')
            logging.debug(f'self.weights.mean(): {self.weights.mean()}')

    def epoch_update(self):
        # adjust gamma according to min(weight)

        logging.debug('Weights: {}'.format(self.weights.reshape(-1)))

        nonzeros = []
        self.min_weight = torch.inf
        for i in range(self.weights.shape[0]):
            if self.weights[i] < 0.0:
                self.weights[i] = 1.0
                self.zero_list.append(i)
            else:
                nonzeros.append(i)
                if self.weights[i] < self.min_weight:
                    self.min_weight = self.weights[i]

        # self.weights[nonzeros] /= self.min_weight
        self.attack_gamma = (1.0 / self.min_weight).data


model = IRNet_intorch(train_dataset.dim_x).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.network_lr)

logging.debug(config.model.show_grad_layers(model))


method = StableAL(model, train_dataset.dim_x, config.sal)


# warnings.filterwarnings("ignore")

end_flag = False

# Create a dictionary to store the best loss for each dataset
best_loss_dict = {}
all_loss_dict = {}


with prog.Progress(
    prog.TextColumn('[progress.description]{task.description}'),
    prog.TextColumn('Loss: {task.fields[annot_val]}'),
    prog.BarColumn(80, 'light_pink3', 'deep_sky_blue4', 'green'),
    prog.MofNCompleteColumn(),
    prog.TimeElapsedColumn(),
    prog.TimeRemainingColumn(),
    prog.SpinnerColumn(),
    refresh_per_second=3,
    disable=not config.cli.show_progress,
) as progress:
    model.train()
    partial_losses = progress.add_task(
        '[deep_pink3] Partial [/deep_pink3]',
        total=config.max_epochs,
        annot_val=' ',
    )
    network_losses = progress.add_task('Training', total=config.max_epochs, annot_val=' ')

    for epoch in range(config.max_epochs):
        train_loss = 0.0

        partial_train_loss = 0.0
        total_train_loss = 0.0

        minima = []

        logging.debug(f'[dark_slate_gray3] Epoch {epoch} [/]', extra=dict(markup=True))
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            loss = config.loss_criterion(outputs, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            partial_train_loss += loss.cpu().item()
            if epoch < config.pre_adv_epochs:
                continue

            if adv_loader is None:
                adv_data, adv_target = data, target
            else:
                adv_data, adv_target = next(iter(adv_loader))
            method.adv_step_new(adv_data, adv_target, end_flag, epoch, batch_idx)

            if config.cli.show_cuda_memory:
                logging.debug('Memory after adv_step():')
                log_cuda_mem()

            # TODO can this if ever actually not be true?
            if epoch >= config.pre_adv_epochs:
                if batch_idx == config.partial.partial_batch:  # train partial trainset
                    break
            else:
                raise ValueError('Whoops')

        partial_train_loss /= batch_idx + 1
        if epoch >= config.pre_adv_epochs:
            method.epoch_update()

        progress.update(
            partial_losses,
            advance=1,
            annot_val=f'{partial_train_loss:.4f}',
        )

        if epoch % config.partial.epochs_between_regens == 0:
            with torch.no_grad():
                logging.debug('sorting training set')
                # for sorting Training set
                sort_MAE = []
                model.eval()

                train_losses = []
                for i, (x, y) in enumerate(train_loader):
                    output = model(x)
                    loss = config.loss_criterion(output, y, reduction='none')
                    train_losses.append(loss)

                    debug_summarize(loss=loss, output=output, x=x, y=y)

                    row = pd.DataFrame(to_np(x), columns=input_cols)
                    row['label'] = to_np(y)
                    row['yhat'] = to_np(output)
                    row['loss'] = to_np(loss)
                    sort_MAE.append(row)

                train_loss = torch.cat(train_losses).mean().item()

                sort_MAE = pd.concat(sort_MAE)
                if epoch % config.partial_epoch_save == 0:
                    sort_MAE.to_feather(exp_dir / f'sorted/train_set_sorting_{epoch}.feather')

                new_train = sort_MAE.sort_values(by=['loss'], ascending=False)
                new_train_dataset = RecurrentDataset(new_train)
                train_loader = DataLoader(
                    new_train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )

                logging.debug(
                    f'Epoch {epoch+1}/{config.max_epochs} - Training loss: {train_loss:.4f} '
                )
                logging.debug('==' * 20)

        losses_dict = {}

        for batch_idx, (data, target) in enumerate(train_loader):
            outputs = method.model(data)
            loss = config.loss_criterion(outputs, target)
            train_loss += loss.item()
        train_loss /= len(train_loader)

        progress.update(
            network_losses,
            advance=1,
            annot_val=f'{train_loss:.4f}',
        )

        losses_dict['Train'] = train_loss
        losses_dict['Partial_train'] = partial_train_loss

        def calculate_loss(data_loader, model, criterion):
            mean_loss = 0.0
            total_batches = len(data_loader)

            for batch_idx, (data, target) in enumerate(data_loader):
                outputs = model(data)
                loss = criterion(outputs, target)
                mean_loss += loss.item()

            mean_loss /= total_batches
            return mean_loss

        rsplts = [f'Rsplt{i}' for i in range(1, 6) if f'Rsplt{i}' in config.data.test_sets]
        others = config.data.test_sets

        for dataset_name in rsplts + list(others):
            (_dataset, loader) = test_sets[dataset_name]
            loader_mse_loss = calculate_loss(loader, model, config.loss_criterion)
            losses_dict[dataset_name] = loader_mse_loss

        if rsplts:
            losses_dict['rsplt_ave'] = np.average([losses_dict[rsplt] for rsplt in rsplts])

        save = []
        # Separate loop to update the best loss for all datasets
        for dataset_name, loss in losses_dict.items():
            if dataset_name not in best_loss_dict or loader_mse_loss < best_loss_dict[dataset_name]:
                best_loss_dict[dataset_name] = loader_mse_loss
                save.append(dataset_name)

        # Stop the training process if the training loss has stopped decreasing or has started to increase
        if train_loss < train_best_loss or epoch == config.pre_adv_epochs * 2:
            train_best_loss = train_loss
            counter = 0
            torch.save(model, exp_dir / f'models/IR3_epoch_{epoch}.pt')
            torch.save(
                method.weights,
                exp_dir / f'models/SAL_weight_{epoch}_gamma_{method.attack_gamma}.pt',
            )
            save.append('Train')
        else:
            counter += 1
            logging.debug(f'Training Loss has not improved for {counter} epochs.')

        if losses_dict['rsplt_ave'] < Rsplt_ave_best_loss:
            Rsplt_ave_best_loss = losses_dict['rsplt_ave']
            counter_val = 0
            torch.save(method.model, exp_dir / 'models/IR3_SAL-bset-Rsplt_test_mse_loss.pt')
            save.append('Rsplt_AVE')
        else:
            counter_val += 1
            if counter_val >= config.early_stop:
                logging.info(
                    f'Training stopped. Valid (rand) Loss has not improved for {config.early_stop} epochs.'
                )
                break

        all_loss_dict[epoch + 1] = losses_dict

        mse_losses_df = pd.DataFrame.from_dict(all_loss_dict, orient='index')

        mse_losses_df.reset_index().rename(columns={'index': 'epoch'}).to_feather(
            exp_dir / 'SAL-training_loss.feather'
        )


torch.save(method.weights, exp_dir / f'Whole_SAL_{epoch}.pt')
torch.save(method.model, exp_dir / f'IR3_epoch_{epoch}.pt')

torch.cuda.memory._dump_snapshot("snapshot.pickle")