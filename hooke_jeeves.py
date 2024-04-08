import numpy as np
import matplotlib.pyplot as plt

def golden_section(f, y, d_j, a, b, tol):
    l = a + (1-0.618)*(b-a)
    u = a + 0.618*(b-a)
    for _ in range(100):
        if (b-a) < tol:
            break
        elif f(y + l*d_j) > f(y + u*d_j):
            a = l
            l = u
            u = a + 0.618*(b-a)
        elif f(y + l*d_j) < f(y + u*d_j):
            b = u
            u = l
            l = a + (1-0.618)*(b-a)
    return (a+b)/2

def hooke_jeeves(f,x,tol,f_opt):
    n=len(x)
    y=np.copy(x)
    trajectory = [np.copy(x)]
    for _ in range(100):
        d=np.eye(n)
        for j in range(n):
            l=f_opt(f,y,d[j],-5,5,0.1)
            #print(k,x,j,y,d,l)
            y += l*d[j]
            trajectory.append(np.copy(y))
        if np.linalg.norm(y - x) < tol:
            break
        else:
            d = y-x 
            x = np.copy(y)
            l = f_opt(f,x,d,-5,5,0.1)
            y = x + l*d
    return y, _, trajectory

def f(x):
    return (x[0]-2)**4 + (x[0]-2*x[1])**2

x0 = np.array([0.0,3.0])
x_opt, k, trajectory= hooke_jeeves(f, x0, 0.05, golden_section)
print("x =", x_opt, "-->","f(x) =", f(x_opt))
print(k, "Iterations")

# Graficar la función 
x = np.linspace(-1, 4, 400)
y = np.linspace(-1, 4, 400)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour plot of f(x)')
plt.grid(True)

trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', label='Trajectory')
plt.plot(x0[0],x0[1], 'bo', label='Initial Point') 
plt.plot(x_opt[0], x_opt[1], 'go', label='Optimal Point')
plt.legend()
plt.show()