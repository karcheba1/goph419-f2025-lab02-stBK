import numpy as np

from lab02.linalg import (
    forward_substitution,
)


def main():
    print("Hi! I am in main().")
    print(f"np.__name__: {np.__name__}")

    test_forward_substitution()


def test_forward_substitution():
    print("\nTesting forward substitution:")
    # lower triangular coefficient matrix
    A = np.array(
        [
            [
                2135.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                -2135.0,
                5200.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                -5200.0,
                5796.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                -5796.0,
                7060.0,
            ],
        ]
    )
    print("A:")
    print(A)

    # right-hand side vector
    b = np.array([500.0, 700.0, 1000.0, 500.0])
    print("b:")
    print(b)

    # solve using numpy.linalg.solve()
    x_exp = np.linalg.solve(A, b)
    print("expected (numpy.linalg.solve):")
    print(x_exp)
    print(f"expected shape: {x_exp.shape}")

    # solve using lab02.linalg.forward_substitution()
    x_act = forward_substitution(A, b)
    print("actual (lab02.linalg.forward_substitution):")
    print(x_act)
    print(f"actual shape: {x_act.shape}")


if __name__ == "__main__":
    main()
