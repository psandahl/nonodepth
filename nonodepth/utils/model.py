from __future__ import annotations

from prettytable import PrettyTable
from torch.nn import Module


def total_parameters(model: Module) -> int:
    """
    Count the total number of parameters in the given model.

    Parameters:
        model: A pytorch model.

    Returns:
        The number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def trainable_parameters(model: Module) -> int:
    """
    Count the number of trainable parameters in the given model.

    Parameters:
        model: A pytorch model.

    Returns:
        The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_parameters(model: Module) -> None:
    """
    Print a table of modules and their trainable parameter count.

    Parameters:
        model: A pytorch model.
    """
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f'Total trainable parameters= {total_params}')
