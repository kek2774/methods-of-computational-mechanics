import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from progonka_matrix import clean_solve_progonka_matrix

alpha: np.ndarray = np.array([1, 0.5])
beta: np.ndarray = np.array([0, 1])
gamma: np.ndarray = np.array([0.7, 1])
a: float = 0.6
b: float = 0.9
N: int = 10

h: float = (b - a) / N
nodes: np.ndarray = np.array([a + n * h for n in range(N + 1)])


def p(x: np.ndarray) -> np.ndarray:
    """
    Функция, вычисляющая значение функции перед первой производной в точке x

    Args:
        x (np.ndarray): точка, в которой ищется значение функции

    Returns:
        np.ndarray: значение функции в точке x
    """

    return np.full_like(x, 2, dtype=float)


def q(x: np.ndarray) -> np.ndarray:
    """
    Функция, вычисляющая значение функции перед нулевой производной в точке x

    Args:
        x (np.ndarray): точка, в которой ищется значение функции

    Returns:
        np.ndarray: значение функции в точке x
    """

    return -x


def f(x: np.ndarray) -> np.ndarray:
    """
    Функция, вычисляющая значение функции в правой части уравнения в точке x

    Args:
        x (np.ndarray): точка, в которой ищется значение функции

    Returns:
        np.ndarray: значение функции в точке x
    """
    return np.square(x)


# Для решения прогонкой
A: np.ndarray = np.zeros((N + 1, N + 1), dtype=float)
b_vec: np.ndarray = np.zeros(N + 1, dtype=float)

# Первая строка k = 0
A[0, 0:2] = [-alpha[0] + beta[0] * h, alpha[0]]
b_vec[0] = gamma[0] * h


# Значения функций во всех узлах
p_arr: np.ndarray = p(nodes)
q_arr: np.ndarray = q(nodes)
f_arr: np.ndarray = f(nodes)

# Внутренние узлы k = 1..N-1
for k in range(1, N):
    h_sq = np.pow(h, 2)
    p_k, q_k, f_k = p_arr[k], q_arr[k], f_arr[k]
    A[k, k - 1 : k + 2] = [1 - (p_k * h) / 2, -2 + q_k * h_sq, 1 + p_k * h / 2]
    b_vec[k] = f_k * h_sq

# Строка N
A[N, N - 1 : N + 1] = [-alpha[1], alpha[1] + beta[1] * h]
b_vec[N] = gamma[1] * h

result = clean_solve_progonka_matrix(A, b_vec)
x_vec = result[0]  # type: ignore
residuals_vec = result[1]  # type: ignore
print(np.round(x_vec, 3))
print(residuals_vec)


# Красивый график

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    }
)

fig, ax = plt.subplots(figsize=(11, 6), dpi=200)

fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

blue = "#2563EB"
dark = "#172033"
gray = "#64748B"


# Основная линия
ax.plot(
    nodes,
    x_vec,
    color=blue,
    linewidth=3,
    solid_capstyle="round",
    zorder=3,
    path_effects=[
        pe.SimpleLineShadow(offset=(0, -2), alpha=0.12),
        pe.Normal(),
    ],
)

if len(nodes) <= 15:
    ax.scatter(
        nodes,
        x_vec,
        s=45,
        color=blue,
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )

# Подписи осей
ax.set_xlabel("$x$", fontsize=15, color=dark, labelpad=12)
ax.set_ylabel("$y(x)$", fontsize=15, color=dark, labelpad=12)

# Если точек много — не подписываем каждый x, matplotlib сам выберет норм
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

# Сетка
ax.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    color="#CBD5E1",
    alpha=0.65,
)

ax.set_axisbelow(True)

# Убираем лишние рамки
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for side in ("left", "bottom"):
    ax.spines[side].set_color("#CBD5E1")

ax.tick_params(
    axis="both",
    colors=gray,
    labelsize=11,
    length=0,
    pad=9,
)

# Отступы
ax.margins(x=0.03, y=0.10)

fig.subplots_adjust(
    left=0.10,
    right=0.96,
    bottom=0.14,
    top=0.80,
)

# Сохранение
fig.savefig("boundary_solution.png", dpi=400, bbox_inches="tight", pad_inches=0.25)
fig.savefig("boundary_solution.pdf", bbox_inches="tight", pad_inches=0.25)

plt.show()
