import torch


def load_module_state(module, checkpoint_path, device):
    """Load a state dictionary from ``checkpoint_path``."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    module.load_state_dict(state_dict)
