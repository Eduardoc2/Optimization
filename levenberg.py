import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2

def gradiente(f, x, eps=1e-4):
    gradient = np.zeros(len(x))
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return gradient

def hessian_f(f, x, eps=1e-4):
    hess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                hess[i, i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / eps**2
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
                hess[i, j] = (f(x_plus_plus) - f(x_plus_minus) - f(x_minus_plus) + f(x_minus_minus)) / (4 * eps**2)
    return hess

def levenberg_marquardt(f, x0, tol=1e-6, max_iter=100, lambda_init=0.01, mu=2.0):
    x = x0.copy()
    lambda_val = lambda_init
    trajectory = [x.copy()]
    for _ in range(max_iter):
        grad = gradiente(f,x)
        hess = hessian_f(f,x)
        
        # Calculando el paso de actualización utilizando el método de Levenberg-Marquardt
        step = np.linalg.solve(hess + lambda_val * np.eye(len(x)), -grad)
        
        # Actualizando los parámetros
        x_new = x + step
        
        # Calculando el cambio relativo para verificar convergencia
        rel_change = np.linalg.norm(step) / np.linalg.norm(x)
        
        # Si el cambio relativo es menor que la tolerancia, terminar el algoritmo
        if rel_change < tol:
            break
        
        # Evaluando la función en el nuevo punto
        f_val_new = f(x_new)
        
        # Calculando la mejora esperada y real
        predicted_improve = grad.dot(step)
        actual_improve = f(x) - f_val_new
        
        # Actualizando el parámetro lambda
        if actual_improve > 0:
            ratio = actual_improve / predicted_improve
            if ratio < 0.25:
                lambda_val *= mu
            elif ratio > 0.75:
                lambda_val /= mu
        
        # Actualizando el punto
        trajectory.append(x.copy())
        x = x_new
    return x, _,trajectory

# Punto inicial
x0 = np.array([0.0, 3.0])

# Ejecutando el algoritmo
x_opt, k ,trajectory = levenberg_marquardt(f, x0)

print("x =", x_opt, "-->","f(x) =", f(x_opt))
print(k, "Iterations")

# Graficar la función
x = np.linspace(0, 3, 400)
y = np.linspace(0, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour plot of f(x)')
plt.grid(True)

# Graficar la trayectoria
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', label='Trajectory')
plt.plot(x0[0],x0[1], 'bo', label='Initial Point') 
plt.plot(x_opt[0], x_opt[1], 'go', label='Optimal Point')
plt.legend()
plt.show()