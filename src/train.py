# -*-coding:utf-8-*-
from doctest import debug
import pyrallis
import torch
from torch.autograd import grad
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import csv
import gc
import yaml
import logging
import warnings
import csv

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import os
from datetime import datetime

import rich.progress as prog

from pathlib import Path
from einops import einsum, rearrange, reduce
from config import MainConfig, SALConfig
from utils import debug_cuda, debug_shapes, to_np

config = pyrallis.parse(MainConfig, "configs/defaults.toml")

config.cli.set_up_logging()

torch.autograd.set_detect_anomaly(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_folder_based_on_time(base_path: os.PathLike):
    base_path = Path(base_path)
    # Get the current time
    current_time = datetime.now()

    time_str = current_time.strftime("%m-%d-%H-%M")
    if config.smoke_test:
        time_str = "smoke-test"

    # Create a folder name based on the formatted time
    folder_name = base_path / f"{config.target.col_name}_{time_str}"

    for subfolder in ("adv", "models", "sorted"):
        os.makedirs(folder_name / subfolder, exist_ok=True)

    with (folder_name / "run_params.toml").open("w") as configfile:
        pyrallis.dump(config, configfile)

    return folder_name


exp_dir = create_folder_based_on_time(Path.cwd() / config.log.exp_base_dir)


# # Step 1: Load the CSV file
# inputdataset =pd.read_feather('./nom_dataset_pred.feather')
# #excluded = ["comp",'delta_e', 'bandgap']
# feature_used_columns = inputdataset.columns[:inputdataset.columns.get_loc('TSNE_x')]
# lables = inputdataset.columns[inputdataset.columns.get_loc('Xshift_tsne'):]

# config_dict['dim_x']=len(inputdataset.columns[1:inputdataset.columns.get_loc('bandgap')])


# Xshift_tsne     =inputdataset.loc[inputdataset['Xshift_tsne'    ]==1][feature_used_columns]
# Xshift_umap     =inputdataset.loc[inputdataset['Xshift_umap'    ]==1][feature_used_columns]
# statY_delta_e   =inputdataset.loc[inputdataset['statY_delta_e'  ]==1][feature_used_columns]
# infoY_delta_e   =inputdataset.loc[inputdataset['infoY_delta_e'  ]==1][feature_used_columns]
# statY_bandgap   =inputdataset.loc[inputdataset['statY_bandgap'  ]==1][feature_used_columns]
# infoY_bandgap   =inputdataset.loc[inputdataset['infoY_bandgap'  ]==1][feature_used_columns]
# inRand1         =inputdataset.loc[inputdataset['inRand1'        ]==1][feature_used_columns]
# inRand2         =inputdataset.loc[inputdataset['inRand2'        ]==1][feature_used_columns]
# inRand3         =inputdataset.loc[inputdataset['inRand3'        ]==1][feature_used_columns]
# inRand4         =inputdataset.loc[inputdataset['inRand4'        ]==1][feature_used_columns]
# inRand5         =inputdataset.loc[inputdataset['inRand5'        ]==1][feature_used_columns]
# inPizoe         =inputdataset.loc[inputdataset['inPizoe'        ]==1][feature_used_columns]


# training_set = inputdataset[(inputdataset[lables] == 0).all(axis=1)][feature_used_columns]


# # Step 3: Define a custom PyTorch Dataset class
# class MyDataset(Dataset):
#     def __init__(self, df):
#         self.inputs = df.drop(columns=['delta_e','comp','bandgap']).values
#         self.labels = df['delta_e'].values

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         input = self.inputs[index].tolist()[:]
#         label = self.labels[index].tolist()
#         return torch.tensor(input, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#     def getSALdata(self):
#         input = np.array(self.inputs[:].tolist())
#         label = np.array(self.labels[:].tolist())
#         return (input, label)

# class RecurrentDataset(Dataset):
#     def __init__(self, df):
#         self.inputs = df['data'].values
#         self.labels = df['label'].values

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         input = self.inputs[index].tolist()[:]
#         label = self.labels[index].tolist()
#         return torch.tensor(input, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# # Step 4: Use DataLoader to create batches
# batch_size = config.data.batch_size

# train_dataset = MyDataset(training_set)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)


# Xshift_tsne_loader      =DataLoader(MyDataset(Xshift_tsne  ),batch_size=len(Xshift_tsne  ))
# Xshift_umap_loader      =DataLoader(MyDataset(Xshift_umap  ),batch_size=len(Xshift_umap  ))
# statY_delta_e_loader    =DataLoader(MyDataset(statY_delta_e),batch_size=len(statY_delta_e))
# infoY_delta_e_loader    =DataLoader(MyDataset(infoY_delta_e),batch_size=len(infoY_delta_e))
# statY_bandgap_loader    =DataLoader(MyDataset(statY_bandgap),batch_size=len(statY_bandgap))
# infoY_bandgap_loader    =DataLoader(MyDataset(infoY_bandgap),batch_size=len(infoY_bandgap))
# Rsplt_testset1_loader   =DataLoader(MyDataset(inRand1      ),batch_size=len(inRand1      ))
# Rsplt_testset2_loader   =DataLoader(MyDataset(inRand2      ),batch_size=len(inRand2      ))
# Rsplt_testset3_loader   =DataLoader(MyDataset(inRand3      ),batch_size=len(inRand3      ))
# Rsplt_testset4_loader   =DataLoader(MyDataset(inRand4      ),batch_size=len(inRand4      ))
# Rsplt_testset5_loader   =DataLoader(MyDataset(inRand5      ),batch_size=len(inRand5      ))
# piezo_test_loader       =DataLoader(MyDataset(inPizoe      ),batch_size=len(inPizoe      ))


# data=train_dataset.getSALdata()


# Step 1: Load the CSV file

Rsplt_testset = pd.read_csv("./dataset/Rsplt_testset.csv", index_col=None)
Xshft_testset = pd.read_csv("./dataset/Xshft_testset.csv", index_col=None)
piezo_testset = pd.read_csv("./dataset/piezo_testset.csv", index_col=None)
statY_testset = pd.read_csv("./dataset/statY_testset.csv", index_col=None)
infoY_testset = pd.read_csv("./dataset/infoY_testset.csv", index_col=None)
final_train = pd.read_csv("./dataset/final_trainset.csv", index_col=None)

Rsplt_testset1 = pd.read_csv("./dataset/Rsplt_testset1.csv", index_col=None)
Rsplt_testset2 = pd.read_csv("./dataset/Rsplt_testset2.csv", index_col=None)
Rsplt_testset3 = pd.read_csv("./dataset/Rsplt_testset3.csv", index_col=None)
Rsplt_testset4 = pd.read_csv("./dataset/Rsplt_testset4.csv", index_col=None)
Rsplt_testset5 = pd.read_csv("./dataset/Rsplt_testset5.csv", index_col=None)


# Step 3: Define a custom PyTorch Dataset class
class MyDataset(Dataset):
    def __init__(self, df):
        self.inputs = df.drop(columns=["delta_e", "pretty_comp"]).values.astype(
            np.float32
        )
        self.labels = df["delta_e"].values.astype(np.float32)
        if config.smoke_test:
            self.inputs = self.inputs[:47]
            self.labels = self.labels[:47]

    @property
    def dim_x(self) -> int:
        return self.inputs.shape[1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        return torch.tensor(input, dtype=torch.float32).to(device), torch.tensor(
            label, dtype=torch.float32
        ).to(device).unsqueeze(0)

    def getSALdata(self):
        input = np.array(self.inputs[:].tolist())
        label = np.array(self.labels[:].tolist())
        return (input, label)


class RecurrentDataset(Dataset):
    def __init__(self, df):
        self.inputs = df["data"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # input = self.inputs[index].tolist()[:]
        # label = self.labels[index].tolist()
        # return torch.tensor(input, dtype=torch.float32).to(device), torch.tensor(
        #     label, dtype=torch.float32
        # ).to(device).unsqueeze(0)
        input = self.inputs[index]
        label = self.labels[index]
        return torch.tensor(input, dtype=torch.float32).to(device), torch.tensor(
            label, dtype=torch.float32
        ).to(device)


# Step 4: Use DataLoader to create batches
batch_size = config.data.batch_size

train_dataset = MyDataset(final_train)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


Rsplt_test_dataset = MyDataset(Rsplt_testset)
Xshft_test_dataset = MyDataset(Xshft_testset)
piezo_test_dataset = MyDataset(piezo_testset)
statY_test_dataset = MyDataset(statY_testset)
infoY_test_dataset = MyDataset(infoY_testset)

Rsplt_testset1_dataset = MyDataset(Rsplt_testset1)
Rsplt_testset2_dataset = MyDataset(Rsplt_testset2)
Rsplt_testset3_dataset = MyDataset(Rsplt_testset3)
Rsplt_testset4_dataset = MyDataset(Rsplt_testset4)
Rsplt_testset5_dataset = MyDataset(Rsplt_testset5)

Rsplt_testset1_loader = DataLoader(
    Rsplt_testset1_dataset, batch_size=len(Rsplt_testset1)
)
Rsplt_testset2_loader = DataLoader(
    Rsplt_testset2_dataset, batch_size=len(Rsplt_testset2)
)
Rsplt_testset3_loader = DataLoader(
    Rsplt_testset3_dataset, batch_size=len(Rsplt_testset3)
)
Rsplt_testset4_loader = DataLoader(
    Rsplt_testset4_dataset, batch_size=len(Rsplt_testset4)
)
Rsplt_testset5_loader = DataLoader(
    Rsplt_testset5_dataset, batch_size=len(Rsplt_testset5)
)

Rsplt_test_loader = DataLoader(Rsplt_test_dataset, batch_size=len(Rsplt_testset))
Xshft_test_loader = DataLoader(Xshft_test_dataset, batch_size=len(Xshft_testset))
piezo_test_loader = DataLoader(piezo_test_dataset, batch_size=len(piezo_testset))
statY_test_loader = DataLoader(statY_test_dataset, batch_size=len(statY_testset))
infoY_test_loader = DataLoader(infoY_test_dataset, batch_size=len(infoY_testset))


data = train_dataset.getSALdata()


def sample_and_remove_from_testset(test_dataloader, num_samples=config.data.batch_size):
    # Step 1: Get the entire test set
    test_dataset = test_dataloader.dataset
    # TODO set this as config
    seed = 69
    torch.manual_seed(seed)

    # Step 2: Sample specified number of indices from the test set
    sampled_indices = torch.randperm(len(test_dataset))

    # Set seed again to ensure consistency in the order of indices between runs
    torch.manual_seed(seed)
    # Select the first num_samples indices from the shuffled indices
    sampled_indices = sampled_indices[:num_samples]

    # Step 3: Create a DataLoader with the sampled test batch
    sampled_test_dataset = Subset(test_dataset, sampled_indices)
    sampled_test_dataloader = DataLoader(
        sampled_test_dataset, batch_size=num_samples, shuffle=True
    )

    # Step 4: Create a DataLoader with the remaining test set (excluding the sampled indices)
    remaining_indices = [
        i for i in range(len(test_dataset)) if i not in sampled_indices
    ]
    remaining_test_dataset = Subset(test_dataset, remaining_indices)
    remaining_test_dataloader = DataLoader(
        remaining_test_dataset, batch_size=len(remaining_test_dataset), shuffle=True
    )

    # Step 5: Print the lengths of the original test set, sampled test set, and remaining test set
    logging.debug(f"Original Test Set Size: {len(test_dataset)}")
    logging.debug(f"Sampled Test Set Size: {len(sampled_test_dataloader.dataset)}")
    logging.debug(f"Remaining Test Set Size: {len(remaining_test_dataloader.dataset)}")

    return sampled_test_dataloader, remaining_test_dataloader


piezo_adv_loader, piezo1_test_loader = sample_and_remove_from_testset(piezo_test_loader)


with open("feat_col_name.txt", "r") as file:
    column_names = file.read().split()


# column_names = feature_used_columns
def save_tensor(ori_data, ori_lab, adv_data, adv_lab, epoch, batch_idx):
    ori_data_list = ori_data.clone().cpu().numpy()
    ori_labels_list = ori_lab.clone().cpu().numpy()
    adv_data_list = adv_data.clone().cpu().numpy()
    adv_labels_list = adv_lab.clone().cpu().tolist()

    csv_filename = exp_dir / "adv" / f"adv_data_labels_in_attack{epoch}_{batch_idx}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = column_names + [
            "label"
        ]  # [f'feature_{i}' for i in range(ori_data_list.shape[1])]
        writer.writerow(header)

        # Write data and labels for each group
        rows_group1 = np.column_stack((ori_data_list, ori_labels_list))
        rows_group2 = np.column_stack((adv_data_list, ori_labels_list))

        # Combine data from the first two groups only
        all_rows = np.vstack((rows_group1, rows_group2))

        writer.writerows(all_rows)


class IRNet_intorch(torch.nn.Module):
    #'128-64-16'
    def __init__(self, input_size):
        super(IRNet_intorch, self).__init__()
        self.fc128 = nn.Linear(128, 128)
        self.fc64 = nn.Linear(64, 64)
        self.fc16 = nn.Linear(16, 16)


        if
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


train_best_loss = float("inf")
partial_train_best_loss = float("inf")

Rsplt_ave_best_loss = float("inf")

# Xshift_tsne_best_loss = float('inf')
# Xshift_umap_best_loss = float('inf')
# statY_delta_e_best_loss = float('inf')
# infoY_delta_e_best_loss = float('inf')
# statY_bandgap_best_loss = float('inf')
# infoY_bandgap_best_loss = float('inf')
# Rsplt_testset1_best_loss = float('inf')
# Rsplt_testset2_best_loss = float('inf')
# Rsplt_testset3_best_loss = float('inf')
# Rsplt_testset4_best_loss = float('inf')
# Rsplt_testset5_best_loss = float('inf')
# piezo_test_best_loss = float('inf')

loss_df = pd.DataFrame(
    columns=[
        "epoch",
        "train",
        "partialtrain",
        "Xshift_tsne",
        "Xshift_umap",
        "statY_delta_e",
        "infoY_delta_e",
        "statY_bandgap",
        "infoY_bandgap",
        "inRand1",
        "inRand2",
        "inRand3",
        "inRand4",
        "inRand5",
        "inPizoe",
        "RspltAVE",
        "save",
        "attack_gamma",
    ]
)


def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()

    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


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
    def __init__(self, model: torch.nn.Module, dim_x: int, sal_conf: SALConfig):
        self.weights = None
        self.model = model
        self.weight_grad = None
        self.xa_grad = None
        self.theta_grad = None
        self.gamma = None
        self.adversarial_data = None
        self.loss_criterion = torch.nn.MSELoss()

        self.conf = sal_conf
        self.adv_based_on = None
        self.adv_again = None

        self.X = None
        self.y = None

        self.dim_x = dim_x

        # self.model = IRNet_intorch(dim_x).to(device)
        # # Covariate Weights
        self.weights = torch.zeros(dim_x).reshape(-1, 1) + 100.0
        self.weights = self.weights.to(device)

    def cost_function(self, x, x_adv):
        # Variable cost level where the weights determine the cost level
        cost = torch.mean(((x - x_adv) ** 2).mm(self.weights)).to(device)
        return cost

    # Loss across Training environments
    # Self.loss_criterion = MSELoss
    def r(self, environments, alpha=None):
        result = 0.0
        env_loss = []
        for x_e, y_e in environments:
            x_e = x_e.to(device)
            y_e = y_e.to(torch.float32).to(device)
            env_loss.append(self.loss_criterion(self.model(x_e), y_e))
        env_loss = torch.Tensor(env_loss)
        max_index = torch.argmax(env_loss)
        min_index = torch.argmin(env_loss)

        for idx, (x_e, y_e) in enumerate(environments):
            x_e = x_e.to(device)
            y_e = y_e.to(torch.float32).to(device)
            if idx == max_index:
                result += (alpha + 1) * self.loss_criterion(self.model(x_e), y_e)
            elif idx == min_index:
                result += (1 - alpha) * self.loss_criterion(self.model(x_e), y_e)
            else:
                result += self.loss_criterion(self.model(x_e), y_e)
        return result

    # generate adversarial data
    # Maximize the loss using their own ADAM.update method(their own optimizer)
    def attack(self, gamma, data, epoch, batch_idx):
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
            loss = self.loss_criterion(outputs, labels) - gamma * self.cost_function(
                images, images_adv
            )

            loss.backward()

            images_adv.data = optimizer.update(images_adv.grad, i + 1, images_adv)

        self.weight_grad = -2 * gamma * attack_lr * (images_adv - images)
        temp_image = images_adv.clone().detach()
        temp_label = labels.clone().detach()
        self.adversarial_data = (temp_image, temp_label)

        # save adv and ori data
        save_tensor(images, labels, temp_image, temp_label, epoch, batch_idx)

        return images_adv, labels

    # Optimizes the model paremeters such that the loss is minimized
    # on the adversarial data from self.attack
    def train_theta(
        self,
        model,
        data,
        gamma,
        end_flag=False,
        epoch=0,
        batch_idx=None,
    ):
        optimizer = optim.Adam(model.parameters(), lr=self.conf.theta_lr)
        self.adv_based_on = data
        # For __ Theta self.conf.theta_epochs
        for i_theta in range(self.conf.theta_epochs):
            if i_theta % self.conf.adv_reset_epochs == 0 or not end_flag:
                images_adv, labels = self.attack(
                    gamma, data, epoch=epoch, batch_idx=batch_idx
                )

            else:
                self.adv_again = self.adversarial_data
                images_adv, labels = self.attack(
                    gamma,
                    self.adversarial_data,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )

            # TODO deal with batch norm properly
            # from torch.func import replace_all_batch_norm_modules_
            # replace_all_batch_norm_modules_(self.model)

            optimizer.zero_grad()
            images_adv = images_adv.to(device)
            outputs = model(images_adv)
            loss = self.loss_criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            assert len(list(model.parameters())) == len(list(model.named_parameters()))

            nuisance_params = {k: v.detach() for k, v in model.named_parameters()}
            assert self.conf.grad_layer in nuisance_params
            grad_param = nuisance_params.pop(self.conf.grad_layer)

            from torch.func import vmap, jacfwd, grad as funcgrad

            # debug_shapes("images_adv", "labels", **nuisance_params)

            def base_loss(grad_param, inputs, targets, nuisance_params):
                nuisance_params[self.conf.grad_layer] = grad_param.reshape(16, 16)
                outputs = torch.func.functional_call(model, nuisance_params, inputs)
                loss = self.loss_criterion(outputs, targets)
                return loss

            dl_th = jacfwd(base_loss)

            # l_th_out = dl_th(grad_param, images_adv, labels, nuisance_params)
            # debug_shapes("l_th_out")

            d2l_th_x = jacfwd(dl_th, argnums=1)

            dtheta_dx = d2l_th_x(grad_param, images_adv, labels, nuisance_params)

            dtheta_dx = reduce(dtheta_dx, "l1 l2 batch dim_x -> l1 l2 dim_x", "sum")

            self.xa_grad = 0 if self.xa_grad is None else self.xa_grad
            self.xa_grad += dtheta_dx
        self.xa_grad *= self.conf.xa_grad_reduce

    def adv_step(self, model, data, target, attack_gamma, end_flag, epoch, batch_idx):
        # TODO split this from adv_target and adv_data
        self.train_theta(
            model,
            (data, target),
            attack_gamma,
            end_flag,
            epoch,
            batch_idx,
        )

        rtheta = self.r([[data, target]], alpha=self.conf.alpha / math.sqrt(epoch + 1))
        self.theta_grad = grad(
            rtheta,
            list(model.parameters())[4],
            # create_graph=True,
            # allow_unused=True,
        )[0]

        # dR/dθ: l1 l2
        # dθ/dX: l1 l2 dim_x
        # dX/dw: dim_x

        debug_shapes(th_gr=self.theta_grad, xa=self.xa_grad, wg=self.weight_grad)

        deltaw = einsum(
            self.theta_grad,
            self.xa_grad,
            self.weight_grad,
            "l1 l2, l1 l2 dim_x, b dim_x -> dim_x",
        )
        deltaw *= -1

        deltaw[zero_list] = 0.0
        max_grad = torch.max(torch.abs(deltaw))
        deltastep = self.conf.deltaall
        lr_weight = (deltastep / max_grad).detach()
        logging.debug(f"RLoss: {rtheta.data}")

        self.weights -= lr_weight * deltaw.detach().reshape(self.weights.shape)

        # debug_cuda()


model = IRNet_intorch(train_dataset.dim_x).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.network_lr)

method = StableAL(model, train_dataset.dim_x, config.sal)
# TODO in StableAL([train_dataset.getSALdata()]), why was the list being passed in? AFAIK it wasn't
# being used

adv_data, adv_target = next(iter(piezo_adv_loader))


# warnings.filterwarnings("ignore")

min_weight = torch.min(method.weights)
attack_gamma = (1.0 / min_weight).data
zero_list = []
end_flag = False

# Create a dictionary to store the best loss for each dataset
best_loss_dict = {}
all_loss_dict = {}


with prog.Progress(
    prog.TextColumn("[progress.description]{task.description}"),
    prog.TextColumn("Loss: {task.fields[annot_val]}"),
    prog.BarColumn(80, "light_pink3", "deep_sky_blue4", "green"),
    prog.MofNCompleteColumn(),
    prog.TimeElapsedColumn(),
    prog.TimeRemainingColumn(),
    prog.SpinnerColumn(),
    refresh_per_second=3,
    disable=not config.cli.show_progress,
) as progress:
    partial_losses = progress.add_task(
        "[deep_pink3] Partial [/deep_pink3]",
        total=config.max_epochs,
        annot_val=" ",
    )
    network_losses = progress.add_task(
        "Training", total=config.max_epochs, annot_val=" "
    )
    # progress.update(total_epochs, advance=1, annot_name="partial_loss", annot_val="")

    for epoch in range(config.max_epochs):
        train_loss = 0.0

        # Xshift_tsne_mse_loss = 0
        # Xshift_umap_mse_loss = 0
        # statY_delta_e_mse_loss = 0
        # infoY_delta_e_mse_loss = 0
        # statY_bandgap_mse_loss = 0
        # infoY_bandgap_mse_loss = 0
        # Rsplt_testset1_mse_loss = 0
        # Rsplt_testset2_mse_loss = 0
        # Rsplt_testset3_mse_loss = 0
        # Rsplt_testset4_mse_loss = 0
        # Rsplt_testset5_mse_loss = 0
        # piezo_test_mse_loss = 0

        partial_train_loss = 0.0
        total_train_loss = 0.0

        minima = []

        def sizeof_fmt(num, suffix="B"):
            for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
                if abs(num) < 1024.0:
                    return f"{num:3.1f}{unit}{suffix}"
                num /= 1024.0
            return f"{num:.1f}Yi{suffix}"

        for batch_idx, (data, target) in enumerate(train_loader):
            # logging.debug(f"current in epoch    {epoch}      batch {batch_idx}")
            logging.debug(
                f"CUDA allocated: {sizeof_fmt(torch.cuda.memory_allocated())}"
            )

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

            method.adv_step(
                model, data, target, attack_gamma, end_flag, epoch, batch_idx
            )

            # TODO can this if ever actually not be true?
            if epoch >= config.pre_adv_epochs:
                if batch_idx == config.partial.partial_batch:  # train partial trainset
                    break

        partial_train_loss /= batch_idx + 1

        progress.update(
            partial_losses,
            advance=1,
            annot_val=f"{partial_train_loss:.4f}",
        )

        if epoch % config.partial.epochs_between_regens == 0:
            with torch.no_grad():
                logging.debug("sorting training set")
                # for sorting Training set
                sort_MAE = []
                method.model.eval()
                for i, (x, y) in enumerate(train_dataset):
                    # inp = train_dataset.inputs[i]
                    # tar = train_dataset.labels[i]
                    # x = torch.tensor([inp.tolist()], dtype=torch.float32).to(device)
                    # y = torch.tensor(tar.tolist(), dtype=torch.float32).to(device)

                    output = method.model(x.unsqueeze(0))
                    loss = config.loss_criterion(output, y.unsqueeze(0)).cpu()
                    # Accumulate the training loss
                    train_loss += loss.item()
                    # logging.debug(f"loss:       {loss}")
                    sort_MAE.append(
                        {"data": to_np(x), "label": to_np(y), "loss": loss.item()}
                    )

                if epoch % config.partial_epoch_save == 0:
                    sort_MAE = pd.DataFrame(sort_MAE)
                    sort_MAE.to_csv(
                        exp_dir / "sorted/train_set_sorting_{epoch}.csv", index=False
                    )

                sort_MAE = pd.DataFrame(sort_MAE)
                new_train = sort_MAE.sort_values(by=["loss"], ascending=False)
                new_train_dataset = RecurrentDataset(new_train)
                train_loader = DataLoader(
                    new_train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )

                train_loss /= len(train_dataset.inputs)
                logging.debug(
                    f"Epoch {epoch+1}/{config.max_epochs} - Training loss: {train_loss:.4f} "
                )
                logging.debug("==" * 20)

        losses_dict = {}

        for batch_idx, (data, target) in enumerate(train_loader):
            outputs = method.model(data)
            loss = config.loss_criterion(outputs, target)
            train_loss += loss.item()
        train_loss /= len(train_loader)

        progress.update(
            network_losses,
            advance=1,
            annot_val=f"{train_loss:.4f}",
        )

        losses_dict["Train"] = train_loss
        losses_dict["Partial_train"] = partial_train_loss

        def calculate_loss(data_loader, model, criterion):
            mean_loss = 0.0
            total_batches = len(data_loader)

            for batch_idx, (data, target) in enumerate(data_loader):
                outputs = model(data)
                loss = criterion(outputs, target)
                mean_loss += loss.item()

            mean_loss /= total_batches
            return mean_loss

        data_loaders = {
            # 'Xshift_tsne': Xshift_tsne_loader, 'Xshift_umap': Xshift_umap_loader,
            # 'statY_delta_e': statY_delta_e_loader, 'infoY_delta_e': infoY_delta_e_loader,
            # 'statY_bandgap': statY_bandgap_loader, 'infoY_bandgap': infoY_bandgap_loader,
            "Rsplt_testset1": Rsplt_testset1_loader,
            "Rsplt_testset2": Rsplt_testset2_loader,
            "Rsplt_testset3": Rsplt_testset3_loader,
            "Rsplt_testset4": Rsplt_testset4_loader,
            "Rsplt_testset5": Rsplt_testset5_loader,
            "piezo_test": piezo1_test_loader,
        }

        # Iterate over each data loader and calculate MSE loss
        for dataset_name, loader in data_loaders.items():
            loader_mse_loss = calculate_loss(
                loader, method.model, config.loss_criterion
            )
            # Append the MSE loss to the list in the dictionary
            losses_dict[dataset_name] = loader_mse_loss

        rsplt_ave = np.average(
            [
                losses_dict["Rsplt_testset1"],
                losses_dict["Rsplt_testset2"],
                losses_dict["Rsplt_testset3"],
                losses_dict["Rsplt_testset4"],
                losses_dict["Rsplt_testset5"],
            ]
        )
        losses_dict["rsplt_ave"] = rsplt_ave

        save = []
        # Separate loop to update the best loss for all datasets
        for dataset_name, loss in losses_dict.items():
            if (
                dataset_name not in best_loss_dict
                or loader_mse_loss < best_loss_dict[dataset_name]
            ):
                best_loss_dict[dataset_name] = loader_mse_loss
                save.append(dataset_name)

        # Stop the training process if the training loss has stopped decreasing or has started to increase
        if train_loss < train_best_loss:
            train_best_loss = train_loss
            counter = 0
            torch.save(method.model, exp_dir / f"models/IR3_epoch_{epoch}.pt")
            torch.save(
                method.weights,
                exp_dir / "models/SAL_weight_{epoch}_gamma_{attack_gamma}.pt",
            )
            save.append("Train")
        else:
            counter += 1
            logging.debug(f"Training Loss has not improved for {counter} epochs.")

        if rsplt_ave < Rsplt_ave_best_loss:
            Rsplt_ave_best_loss = rsplt_ave
            counter_val = 0
            torch.save(
                method.model, exp_dir / "models/IR3_SAL-bset-Rsplt_test_mse_loss.pt"
            )
            save.append("Rsplt_AVE")
        else:
            counter_val += 1
            if counter_val >= config.early_stop:
                logging.info(
                    f"Training stopped. Valid (rand) Loss has not improved for {config.early_stop} epochs."
                )
                break

        all_loss_dict[epoch + 1] = losses_dict

        mse_losses_df = pd.DataFrame.from_dict(all_loss_dict, orient="index")

        mse_losses_df.reset_index().rename(columns={"index": "epoch"}).to_feather(
            exp_dir / "SAL-training_loss.feather"
        )

        # adjust gamma according to min(weight)

        for i in range(method.weights.shape[0]):
            if method.weights[i] > 0.0 and method.weights[i] < min_weight:
                min_weight = method.weights[i]
            if method.weights[i] < 0.0:
                method.weights[i] = 1.0
                zero_list.append(i)

        attack_gamma = (1.0 / min_weight).data
        if epoch <= config.pre_adv_epochs:
            continue


torch.save(method.weights, exp_dir / f"Whole_SAL_{epoch}.pt")
torch.save(method.model, exp_dir / f"IR3_epoch_{epoch}.pt")
