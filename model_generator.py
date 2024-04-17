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
            l=f_opt(y,d[j],-4,4,0.001,input,output)
            y += l*d[j]
        if np.linalg.norm(y - x0) < tol:
            break
        else:
            x0=np.copy(y)
    return y, _

# Leer el archivo .dat 
datos = np.loadtxt('dryer.dat')

# Number of samples (init, final, 1) valor inicial >2 !!!
n_s=[2,100,1]
time = np.arange(*n_s)

input = datos[n_s[0]:n_s[1],0]
output = datos[n_s[0]:n_s[1],1]
x_prev = datos[n_s[0]-2:n_s[0],0]
y_prev = datos[n_s[0]-2:n_s[0],1]


theta,n_iter=cyclic_coordinate([0.5,0.5,0.1,0.1],golden_section,input,output,tol=0.0001)

print(theta,n_iter)

res=[-1.50679337,  0.57994944, -0.04300483,  0.11451339]
model_out = predict_out(*theta,input)

cos=cost(theta,input,output)
print(cos)
# Graficar cada columna respecto al tiempo

plt.figure(figsize=(10, 6))
plt.plot(time, input, '-s',label='Input')
plt.plot(time, output, '-d',label='Real Output')
plt.plot(time, model_out, '-o', label='Model Output')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Model')
plt.grid(True)
plt.legend()
plt.show()