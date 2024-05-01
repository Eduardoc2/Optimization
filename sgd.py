import numpy as np
import matplotlib.pyplot as plt

def loss_function(x, y):
    return (x - 2)**4 + (x - 2*y)**2

def gradient(f, x, eps=1e-4):
    gradient = np.zeros(len(x))
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        gradient[i] = (f(*x_plus) - f(*x_minus)) / (2 * eps)
    return gradient

def stochastic_gradient_descent(initial_point, batch_size, max_iter, tol):
    point = np.copy(initial_point)
    path = [initial_point]  
    for k in range(max_iter):
        grad= gradient(loss_function,point)
        learning_rate = learning_rate_schedule(k)
        point -= learning_rate * grad
        if np.linalg.norm(grad) < tol:
            return path, k
        path.append(point)
    return path, k

def learning_rate_schedule(k):
    tau = max_iter/2
    e_init = 0.1
    e_tau = 0.01
    a = k / tau
    return (1 - a) * e_init + a * e_tau

initial_point = np.array([0.0, 0.0])  
batch_size = 1  # Batch size (not relevant for this problem, but kept for consistency)
max_iter = 100
tol = 1e-3  
path, k = stochastic_gradient_descent(initial_point, batch_size, max_iter, tol)
print("x =", path[-1], "-->","f(x) =", loss_function(*path[-1]))
print(k, "Iterations")

# Graficar la función
x = np.linspace(0, 3, 400)
y = np.linspace(0, 3, 400)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour plot of f(x)')
plt.grid(True)

path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'ro-', label='Trajectory')
plt.plot(initial_point[0], initial_point[1], 'bo', label='Initial Point') 
plt.plot(*path[-1], 'go', label='Optimal Point')
plt.legend()
plt.show()

