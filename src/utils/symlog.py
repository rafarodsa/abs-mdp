import torch

def symlog(x, c=1.0):
    """Symmetric log function.

    Parameters
    ----------
    x : float
        Input value.
    c : float
        Constant.

    Returns
    -------
    float
        Symmetric log function of x.

    """
    return torch.sign(x) * torch.log(torch.abs(x) + c)

def symexp(x, c=1.0):
    """Symmetric exp function.

    Parameters
    ----------
    x : float
        Input value.
    c : float
        Constant.

    Returns
    -------
    float
        Symmetric exp function of x.

    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - c)

