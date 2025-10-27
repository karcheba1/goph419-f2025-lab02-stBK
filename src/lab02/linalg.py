import numpy as np


def forward_substitution(L, b):
    """Solve a system Lx = b
    where L is a lower triangular coefficient matrix
    and b is the right-hand side vector,
    or a matrix where each column is a right-hand side vector.

    Parameters
    ----------
    L : array_like
        Lower triangular matrix, size = (n, n)
    b : array_like
        Right-hand side(s), size = (n, ) or (n, m)
        where m is the number of right-hand sides.

    Returns
    -------
    numpy.ndarray
        The vector or matrix of solutions x.
        This will have the same shape as b.

    """
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)

    return np.zeros_like(b)
