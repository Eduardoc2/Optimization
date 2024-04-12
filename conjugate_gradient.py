import numpy as np
import matplotlib.pyplot as plt

def golden_section(f, x, d_k, a, b, tol):
    l = a + (1-0.618)*(b-a)
    u = a + 0.618*(b-a)
    for _ in range(100):
        if (b-a) < tol:
            break
        elif f(x + l*d_k) > f(x + u*d_k):
            a = l
            l = u
            u = a + 0.618*(b-a)
        elif f(x + l*d_k) < f(x + u*d_k):
            b = u
            u = l
            l = a + (1-0.618)*(b-a)
    return abs((a+b)/2)

def conjugate_gradient(f,x,tol,f_opt):
    y = np.copy(x)
    d = -gradiente(f,y)
    trajectory = [np.copy(x)]
    for _ in range(100):
        for j in range(len(x)):
            l=f_opt(f,y,d,-5,5,0.1)
            y_new =  y + l*d
            #print("y",y,"d",d,"lambda",l,"y_new",y_new)
            a = np.linalg.norm(gradiente(f,y_new))**2/np.linalg.norm(gradiente(f,y))**2
            d = - gradiente(f,y_new) + a * d
            y =np.copy(y_new)
            trajectory.append(np.copy(y))
        if np.linalg.norm(gradiente(f,y)) < tol:
            break
        d = - gradiente(f,y_new)
        x=np.copy(y)
    return y, _, trajectory
      
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

x0 = np.array([0.00,3.00])
x_init = x0.copy()
x_opt, k, trajectory= conjugate_gradient(f, x0, 0.05,golden_section)
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
plt.plot(x_init[0],x_init[1], 'bo', label='Initial Point') 
plt.plot(x_opt[0], x_opt[1], 'go', label='Optimal Point')
plt.legend()
plt.show()
