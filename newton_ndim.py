import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (x - 2)**4 + (x - 2*y)**2

def gradiente(f, x, eps=1e-4):
    gradient = np.zeros(len(x))
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        gradient[i] = (f(*x_plus) - f(*x_minus)) / (2 * eps)
    return gradient

def hessian(f, x, eps=1e-4):
    hess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                hess[i, i] = (f(*x_plus) - 2 * f(*x) + f(*x_minus)) / eps**2
            else:
                x_plus_plus = x.copy()
                x_plus_minus = x.copy()
                x_minus_plus = x.copy()
                x_minus_minus = x.copy()
                x_plus_plus[i] += eps
                x_plus_plus[j] += eps
                x_plus_minus[i] += eps
                x_plus_minus[j] -= eps
                x_minus_plus[i] -= eps
                x_minus_plus[j] += eps
                x_minus_minus[i] -= eps
                x_minus_minus[j] -= eps
                hess[i, j] = (f(*x_plus_plus) - f(*x_plus_minus) - f(*x_minus_plus) + f(*x_minus_minus)) / (4 * eps**2)
    return hess

def newton_ndim(f, hess, x0, tol, max_iter):
    x = []
    x.append(x0)
    for k in range(max_iter):
        hess = hessian(f,x[k])
        grad = gradiente(f,x[k])
        x.append(x[k] - np.dot(np.linalg.inv(hess),grad))
        if np.linalg.norm(x[k+1] - x[k])  < tol:
            break
    return x, k

x0 = np.array([0.00,3.00])
trajectory, k= newton_ndim(f,hessian,x0,tol=0.05,max_iter=100)
print("x =", trajectory[-1], "-->","f(x) =", f(*trajectory[-1]))
print(k, "Iterations")

# Graficar la función 
x = np.linspace(-1, 4, 400)
y = np.linspace(-1, 4, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour plot of f(x)')
plt.grid(True)

# Graficar la trayectoria
trajectory = np.array(trajectory)
print(trajectory[-1])
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', label='Trajectory')
plt.plot(x0[0],x0[1], 'bo', label='Initial Point') 
plt.plot(*trajectory[-1], 'go', label='Optimal Point')
plt.legend()
plt.show()