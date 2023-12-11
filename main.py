import numpy as np
from matplotlib import pyplot as plt, animation
import tqdm

dx = 0.05
dt = 0.002
x = np.arange(-5, 50 + dx, dx)
dk = 1.5
imaginary = bool(1)


def psi0(x):
    k0 = 10
    return np.sqrt(dk) * np.exp(-x ** 2 * dk ** 2 / 2) * np.exp(1j * k0 * x) / np.pi ** (1 / 4)


def V(x):
    param = 10
    sigma = 0.5
    V0 = 105
    return V0 * np.exp(-(x - param) ** 2 / (sigma ** 2))


def get_diagonal_matrix(b, a_c):
    return np.diag(a_c, -1) + np.diag(b, 0) + np.diag(a_c, 1)


def r_vec_i(psi_jm1, psi_j, psi_jp1, vj):
    return psi_j + 1j * dt * ((psi_jp1 - 2 * psi_j + psi_jm1) / (dx ** 2) + vj * psi_j) / 2


def r_vec(psi, v):
    return [r_vec_i(p_jm1, p_j, p_jp1, vj) for p_jm1, p_j, p_jp1, vj in zip(psi[:-2], psi[1:-1], psi[2:], v[1:-1])]


def animate_wave(index):
    return ln1.set_ydata(np.array(np.absolute(psi_s[index]) ** 2))


psiStart = psi0(x)

b = 1 + 1j * dt * (2 / (dx ** 2) + V(x)[1:-1]) / 2
a_c = -1j * dt * np.ones(len(b) - 1) / (2 * dx ** 2)

AMatrix = get_diagonal_matrix(b, a_c)
rVec = r_vec(psiStart, V(x))

J = 1000
psi_i = psi0(x)
psi_s = []
for i in tqdm.tqdm(range(J)):
    psi_i = np.linalg.solve(AMatrix, rVec)
    psi_s.append([0, *psi_i, 0])
    rVec = r_vec([0, *psi_i, 0], V(x))


fig, ax = plt.subplots()
ln1, = ax.plot(x.real, np.absolute(psi0(x))**2, 'r-')
ln2 = ax.plot(x, V(x)/100)
ax.set_ylim(-1, 2)
ax.set_xlim(-5, 50)
title = 'PLot for dk = ' + str(dk) + ', ' + (str('real') if imaginary == 0 else str('imaginary')) + ' x axis values and without potential V'
plt.title(title)
plt.tight_layout()
ani = animation.FuncAnimation(fig, animate_wave, frames=J, interval=1)

ani.save(title + '.gif', writer='pillow', fps=30, dpi=200)
