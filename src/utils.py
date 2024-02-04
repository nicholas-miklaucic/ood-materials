import inspect
import logging

import numpy as np
from tensorboard import summary
import torch


def to_np(x: torch.Tensor) -> np.ndarray:
    """Converts to numpy."""
    return x.detach().cpu().numpy()


def debug_shapes(*args, **kwargs):
    """Shows the shapes of PyTorch tensor inputs. Kwargs are instead key-value pairs."""
    names = list(args)
    frame = inspect.currentframe().f_back.f_locals
    shapes = [frame[name].shape for name in names]

    for k, v in kwargs.items():
        names.append(k)
        shapes.append(v.shape)

    try:
        max_len = int(max(map(len, shapes)))
        max_digits = len(str(max(map(max, shapes))))
        max_name_len = max(len(name) for name in names)
        for name, shape in zip(names, shapes):
            logging.debug(
                f'{name:>{max_name_len}} = '
                + ' '.join(
                    [' ' * max_digits] * (max_len - len(shape))
                    + [f'{dim:>{max_digits}}' for dim in shape]
                )
            )
    finally:
        del frame


def debug_cuda():
    import gc
    import operator as op
    from functools import reduce

    import torch

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(
                    reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0,
                    str(type(obj)).rsplit('.', 1)[1].removesuffix("'>"),
                    ' '.join(map(str, obj.size())),
                )
        except:
            pass


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.0f} {unit}{suffix}'
        num /= 1024
    return f'{num:.0f}Yi{suffix}'


def pretty(vector):
    if isinstance(vector, list):
        vlist = vector
    elif isinstance(vector, np.ndarray):
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()

    return '[' + ', '.join('{:+.4f}'.format(vi) for vi in vlist) + ']'


def structure(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    elif isinstance(obj, np.ndarray):
        return summary_stat(torch.tensor(obj))
    elif isinstance(obj, (list, tuple)):
        return list(map(structure, obj))
    elif isinstance(obj, (float, int)):
        return 'scalar'
    elif isinstance(obj, dict):
        return {k: structure(v) for k, v in obj.items()}
    else:
        return str(type(obj))


def summary_stat(obj):
    if isinstance(obj, torch.Tensor):
        flat = obj.flatten()
        inds = torch.cos(torch.arange(len(flat), dtype=obj.dtype, device=obj.device))
        return torch.lerp(flat, inds, 0.1).mean().item()
    elif isinstance(obj, np.ndarray):
        return summary_stat(torch.tensor(obj))
    elif isinstance(obj, (list, tuple)):
        return list(map(structure, obj))
    elif isinstance(obj, (float, int)):
        return 'scalar'
    elif isinstance(obj, dict):
        return {k: summary_stat(v) for k, v in obj.items()}
    else:
        return str(type(obj))


def debug_summarize(show_stat=False, **kwargs):
    import logging

    for k, v in kwargs.items():
        logging.debug(f'{k:>30} structure:\t{structure(v)}')

        if show_stat:
            logging.debug(f'{k:>30} stat:     \t{summary_stat(v)}')


def same_storage(x, y):
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


def log_cuda_mem():
    import logging

    logging.debug(f'CUDA allocated: {sizeof_fmt(torch.cuda.memory_allocated())}')
