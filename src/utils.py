import inspect
import logging
import torch
import numpy as np


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
                f"{name:>{max_name_len}} = "
                + " ".join(
                    [" " * max_digits] * (max_len - len(shape))
                    + [f"{dim:>{max_digits}}" for dim in shape]
                )
            )
    finally:
        del frame


def debug_cuda():
    import torch
    import gc
    from functools import reduce
    import operator as op

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(
                    reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0,
                    str(type(obj)).rsplit(".", 1)[1].removesuffix("'>"),
                    " ".join(map(str, obj.size())),
                )
        except:
            pass
