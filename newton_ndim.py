import numpy as np

def funcion(x):
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2

def gradiente(x):
    df_dx = 4 * (x[0] - 2)**3 + 2 * (x[0] - 2*x[1])
    df_dy = -4 * (x[0] - 2*x[1])
    return np.array([df_dx, df_dy])

def hessiana(x):
    d2f_dx2 = 12 * (x[0] - 2)**2 + 2
    d2f_dy2 = 4
    d2f_dxdy = -4
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

def metodo_newton(f, grad, hess, x0, tol, max_iter):
    x = []
    x.append(x0)
    for k in range(max_iter):
        gradd = grad(x[k])
        hessi = hess(x[k])
        x.append(x[k] - np.dot(np.linalg.inv(hessi),gradd))
        if np.linalg.norm(x[k+1] - x[k])  < tol:
            break
    return x[-1]

x0 = np.array([0.0,3.0])
solucion = metodo_newton(funcion, gradiente, hessiana, x0,tol=1e-6, max_iter=100)
print("El valor óptimo de la función es:", solucion)
print("El valor de la función en el punto óptimo es:", funcion(solucion))