import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def f(x):
    return x**2 * np.sin(x)

a, b = -np.pi, np.pi
n = 11
x_nodes = np.linspace(a, b, n)
y_nodes = f(x_nodes)

def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:] = (coef[j:] - coef[j-1:-1]) / (x[j:] - x[:-j])
    return coef

def newton_poly(x_data, y_data, x):
    coef = divided_diff(x_data, y_data)
    n = len(coef)
    result = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        result += term
    return result

cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')

x_dense = np.linspace(a, b, 500)
y_true = f(x_dense)
y_newton = [newton_poly(x_nodes, y_nodes, xi) for xi in x_dense]
y_spline = cs(x_dense)

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true, 'k-', label='Аналітична функція')
plt.plot(x_dense, y_newton, 'r--', label='Поліном Ньютона')
plt.plot(x_dense, y_spline, 'b:', label='Кубічний сплайн')
plt.plot(x_nodes, y_nodes, 'go', label='Вузли інтерполяції')
plt.title('Інтерполяція функції $f(x) = x^2 \\sin x$ на [$-\\pi, \\pi$]')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

y_err_newton = np.abs(y_true - y_newton)
y_err_spline = np.abs(y_true - y_spline)

plt.figure(figsize=(10, 5))
plt.plot(x_dense, y_err_newton, 'r--', label='Похибка Ньютона')
plt.plot(x_dense, y_err_spline, 'b:', label='Похибка сплайна')
plt.title('Графік похибки інтерполяції')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend()
plt.show()
