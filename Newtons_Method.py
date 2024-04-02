def f(x): #Objective function
    return x**2+2*x
def df(x):
    return 2*x+2
def d2f(x):
    return 2

def newton_optimization(f,df,d2f,x0,tol=1e-6,max_iter=100):
    x=[x0]
    for k in range(max_iter):
        x.append(x[k]-df(x[k])/d2f(x[k])) 
        if abs(x[k+1] - x[k])  < tol:
            break
    return k, x[-1]

k, x_min= newton_optimization(f, df, d2f,x0=2)

print("x =", x_min, "-->","f(x) =", f(x_min))
print(k, "Iterations")