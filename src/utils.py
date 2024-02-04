import inspect
import logging

import numpy as np
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
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


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
        return str(list(obj.shape))
    elif isinstance(obj, (list, tuple)):
        return '[{}]'.format(', '.join(map(structure, obj)))
    elif isinstance(obj, (float, int)):
        return 'scalar'
    else:
        return str(type(obj))


def log_cuda_mem():
    import logging

    logging.debug(f'CUDA allocated: {sizeof_fmt(torch.cuda.memory_allocated())}')
