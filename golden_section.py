def f(x):
    return x**2+2*x

def golden_section(f, a, b, tol): 
    l = a + (1-0.618)*(b-a)
    u = a + 0.648*(b-a)
    for k in range(100):
        if (b-a) < tol:
            break
        elif f(l) > f(u):
            a = l
            l = u
            u = a + 0.618*(b-a)
        elif f(l) < f(u):
            b = u
            u = l
            l = a + (1-0.618)*(b-a)
    return a, b, k
            
a, b, k = golden_section(f,-5,5,0.5)  

print("Final Interval: [",a,",",b,"]")  
print("x =", (a+b)/2,"--> f(x) =",f((a+b)/2))
print(k,"Iterations")