from sympy import symbols, diff
import numpy as np 

def bisection(f,a, b, l):
    n = np.log(l/(b-a))/np.log(0.5)
    df=diff(f,x)
    for k in range(0, int(np.ceil(n))):
        lambda_k=0.5*(a+b)
        if df.subs(x,lambda_k) == 0:
            return lambda_k, a, b, k
        elif df.subs(x,lambda_k) > 0:
            b = lambda_k
        else:
            a = lambda_k
    lambda_k=0.5*(a+b)
    return lambda_k ,a ,b, k

x=symbols('x')
f=x**2+2*x
x_opt, a, b, k = bisection(f,-3,6,0.2)

print("Final interval: [", a,",",b,"]")
print("x =",x_opt, "-->","f(x) =",f.subs(x,x_opt))
print(k, "Iterations")