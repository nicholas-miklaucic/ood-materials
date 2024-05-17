from pathlib import Path

import pandas as pd
import torch
from rich.progress import track
from train import IRNet_intorch, config, model, test_sets, train_dataset

folder = Path.cwd() / config.log.exp_base_dir / f'delta_e_{config.log.exp_name}'
for ema in (True,):
    model = IRNet_intorch(train_dataset.dim_x, config.model)

    if ema:
        model = torch.load(folder / 'IR3_ema.pt')
    else:
        fn = list(folder.glob('IR3_epoch_*.pt'))
        print(fn)
        model = torch.load(fn[0])

    df = []
    for dataset_name in track(config.data.test_sets):
        (_dataset, loader) = test_sets[dataset_name]
        for batch_idx, (data, targets) in enumerate(loader):
            outputs = model(data)
            for output, target in zip(outputs, targets):
                loss = config.loss_criterion(output, target)
                df.append(
                    {
                        'dataset': dataset_name,
                        'output': output.item(),
                        'target': target.item(),
                        'loss': loss.item(),
                    }
                )

    df = pd.DataFrame(df)
    if ema:
        df.to_feather(folder / 'results_ema.feather')
    else:
        df.to_feather(folder / 'results.feather')
