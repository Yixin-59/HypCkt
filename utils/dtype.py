from contextlib import contextmanager
import torch


@contextmanager
def hyperbolic_dtype(dtype=torch.float64):
    """Context manager: temporarily set default dtype for hyperbolic ops.

    Usage::

        with hyperbolic_dtype():
            # all new tensors created here default to float64
            h = manifold.expmap0(u)
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def to_hyp_dtype(x, dtype=torch.float64):
    """Cast tensor to hyperbolic computation dtype."""
    return x.to(dtype=dtype)


def to_out_dtype(x, dtype=torch.float32):
    """Cast tensor from hyperbolic dtype back to output dtype."""
    return x.to(dtype=dtype)
