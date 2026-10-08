import matplotlib.pyplot as plt
import numpy as np

from progonka_matrix import clean_solve_progonka_matrix

L: int = 5
T: int = 10
N: int = 40  # число шагов разбиения по x
M: int = 80  # число шагов разбиения по t

h: float = L / N
tau: float = T / M

a: float = 10
b: float = 3.0
c: float = 2.0
alpha: np.ndarray = np.array([1, 1])
beta: np.ndarray = np.array([0, 0])


t: np.ndarray = np.array([j * tau for j in range(M + 1)])
x: np.ndarray = np.array([i * h for i in range(N + 1)])

u: np.ndarray = np.zeros((N + 1, M + 1), dtype=float)
A: np.ndarray = np.zeros(N + 1, dtype=float)
B: np.ndarray = np.zeros(N + 1, dtype=float)
C: np.ndarray = np.zeros(N + 1, dtype=float)
D: np.ndarray = np.zeros(N + 1, dtype=float)

progonka_matrix: np.ndarray = np.zeros((N + 1, N + 1), dtype=float)
progonka_vector: np.ndarray = np.zeros(N + 1, dtype=float)


def phi1(t: float) -> float:
    return np.exp(-t / 10)


def phi2(t: float) -> float:
    return np.cos(t / 10)


def psi1(x: float) -> float:
    return x / 5


def psi2(x: float) -> float:
    return x / 10


# первые два слоя
for i in range(N + 1):
    u[i, 0] = psi1(x[i])

for i in range(N + 1):
    u[i, 1] = u[i, 0] + tau * psi2(x[i])


for tj in range(1, M):
    # крайняя левая точка
    A[0] = 0
    B[0] = beta[0] * h - alpha[0]
    C[0] = alpha[0]
    D[0] = h * phi1(t[tj + 1])
    progonka_matrix[0, 0:2] = np.array([B[0], C[0]])
    progonka_vector[0] = D[0]

    # внутренние точки
    for i in range(1, N):
        A[i] = a * tau**2 / h**2 - b * tau**2 / (2 * h)
        # c * tau ** 2, если без тау в квадрате то результат не очень правдоподобный
        B[i] = c * tau**2 - 1 - 2 * a * tau**2 / h**2
        C[i] = a * tau**2 / h**2 + b * tau**2 / (2 * h)
        D[i] = -2 * u[i, tj] + u[i, tj - 1]
        progonka_matrix[i, i - 1 : i + 2] = np.array([A[i], B[i], C[i]])
        progonka_vector[i] = D[i]

    # крайняя правая точка
    A[N] = -alpha[1]
    B[N] = alpha[1] + h * beta[1]
    C[N] = 0
    D[N] = h * phi2(t[tj + 1])
    progonka_matrix[N, N - 1 : N + 1] = np.array([A[N], B[N]])
    progonka_vector[N] = D[N]

    u[:, tj + 1] = clean_solve_progonka_matrix(progonka_matrix, progonka_vector)[0]  # type: ignore


print(np.array([np.round(i, 2) for i in u]).T)

X, TT = np.meshgrid(x, t, indexing="ij")
fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, TT, u, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")

plt.show()
