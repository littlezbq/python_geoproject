import numpy as np


def wybb(A):
    Bs = []
    As = []

    x = A.reshape(1, -1)
    y = np.argsort(x)
    y = np.argsort(y).reshape(A.shape)

    for i in range(A.shape[0] * A.shape[1]):
        a = np.argwhere(y == i)
        b = A[a[0][0], a[0][1]]
        Bs.append(b)
        As.append((a[0][0], a[0][1]))

    Bs.reverse()
    As.reverse()
    return Bs, As


if __name__ == "__main__":
    x = np.array([[1, 4, 2, 8, 5, 7], [5, 4, 3, 7, 9, 6]])
    b, a = wybb(x)

    for number, arg in zip(b, a):
        print(number, '->', arg)
