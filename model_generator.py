import matplotlib.pyplot as plt
import numpy as np

def predict_out(a1,a2,b1,b2,x):
    y = []
    for n in range(n_s[0],n_s[1],1):
        if n - n_s[0] < 2:
            y.append(- a1*y_prev[1] - a2*y_prev[0] + b1*x_prev[1] + b2*x[0])
        else:
            y.append(- a1*y[n-n_s[0]-1] - a2*y[n-n_s[0]-2] + b1*x[n-n_s[0]-1] + b2*x[n-n_s[0]-2])
    return y


def cost(theta,input,output):
    print(output)
    print(predict_out(*theta,input))
    return np.mean((output-predict_out(*theta,input))**2)
        

def golden_section(y, d_j, a, b, tol,input,output):
    l = a + (1-0.618)*(b-a)
    u = a + 0.618*(b-a)
    for _ in range(100):
        if (b-a) < tol:
            break
        elif cost(y + l*d_j,input,output) > cost(y + u*d_j,input,output):
            a = l
            l = u
            u = a + 0.618*(b-a)
        elif cost(y + l*d_j,input,output) < cost(y + u*d_j,input,output):
            b = u
            u = l
            l = a + (1-0.618)*(b-a)
    return (a+b)/2

def cyclic_coordinate(x0,f_opt,input,output,tol):
    d=np.eye(len(x0))
    y=np.copy(x0)
    for _ in range(100):
        for j in range(len(x0)):
            l=f_opt(y,d[j],-5,5,0.1,input,output)
            print(l*d[j])
            y += l*d[j]
        if np.linalg.norm(y - x0) < tol:
            break
        else:
            x0=np.copy(y)
    return y, _

# Leer el archivo .dat 
datos = np.loadtxt('dryer.dat')

# Number of samples (init, final, 1) valor inicial >2 !!!
n_s=[10,15,1]
tiempo = np.arange(*n_s)

input = datos[n_s[0]:n_s[1],0]
output = datos[n_s[0]:n_s[1],1]
x_prev = datos[n_s[0]-2:n_s[0],0]
y_prev = datos[n_s[0]-2:n_s[0],1]


theta,n_iter=cyclic_coordinate([1,1,1,1],golden_section,input,output,tol=0.05)




# Graficar cada columna respecto al tiempo
'''
plt.figure(figsize=(8, 6))
plt.plot(tiempo, datos[:100, 0], label='Columna X')
plt.plot(tiempo, datos[:100, 1], label='Columna Y')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Gráfico de Columnas respecto al Tiempo')
plt.grid(True)
plt.legend()
plt.show()
'''
