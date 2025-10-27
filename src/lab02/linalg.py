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
    # checking that input arguments can be converted to array of float
    # also making a local copy of these arrays
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)

    # check shape of coefficient matrix L
    L_shape = L.shape
    if not len(L_shape) == 2:
        raise ValueError(
            f"Coefficient matrix L has shape {L_shape}, "
            + f"len(L.shape) == {len(L_shape)}, must be 2."
        )
    n = L_shape[0]
    if n != L_shape[1]:
        raise ValueError(f"Coefficient matrix L has shape {L_shape}, must be square.")

    # check shape of the right-hand-side b
    b_shape = b.shape
    if not (len(b_shape) == 1 or len(b_shape) == 2):
        raise ValueError(
            f"Right-hand-side vector b has shape {b_shape}, "
            + f"len(b.shape) == {len(b_shape)}, must be 1 or 2."
        )
    if n != b_shape[0]:
        raise ValueError(
            f"Coefficient matrix L has {n} rows and cols, "
            + f"right-hand-side b has {b_shape[0]} rows, "
            + "dimensions must match."
        )

    # check for b_shape as 1d
    if len(b_shape) == 1:
        b = np.reshape(b, shape=(n, 1))

    # form the augmented matrix
    aug = np.hstack([L, b])

    for k, row in enumerate(L):
        aug[k : k + 1, n:] = (
            aug[k : k + 1, n:] - L[k : k + 1, :k] @ aug[:k, n:]
        ) / aug[k, k]  # row[k] == L[k,k]

    return np.reshape(aug[:, n:], shape=b_shape)
