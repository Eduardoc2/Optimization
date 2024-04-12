import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (x - 2)**4 + (x - 2*y)**2

def grad_f(x, y):
    df_dx = 4 * (x - 2)**3 + 2 * (x - 2*y)
    df_dy = -4 * (x - 2*y)
    return np.array([df_dx, df_dy])

def hessian_f(x, y):
    d2f_dx2 = 12 * (x - 2)**2 + 2
    d2f_dy2 = 8
    d2f_dxdy = -4
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

def levenberg_marquardt(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100, lambda_init=0.01, mu=2.0):
    x = x0.copy()
    lambda_val = lambda_init
    iteration = 0
    trajectory = [x.copy()]
    for _ in range(max_iter):
        grad = grad_f(*x)
        hess = hessian_f(*x)
        
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
        f_val_new = f(*x_new)
        
        # Calculando la mejora esperada y real
        predicted_improve = grad.dot(step)
        actual_improve = f(*x) - f_val_new
        
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
x_opt, k ,trajectory = levenberg_marquardt(f, grad_f, hessian_f, x0)

print("x =", x_opt, "-->","f(x) =", f(*x_opt))
print(k, "Iterations")

# Graficar la función
x = np.linspace(0, 3, 400)
y = np.linspace(0, 3, 400)
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
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', label='Trajectory')
plt.plot(x0[0],x0[1], 'bo', label='Initial Point') 
plt.plot(x_opt[0], x_opt[1], 'go', label='Optimal Point')
plt.legend()
plt.show()