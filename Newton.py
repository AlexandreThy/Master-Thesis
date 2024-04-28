import numpy as np

def newton(f,Df,epsilon,max_iter,X,Y,x0 = np.array([0.8,1.5])):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn,X,Y)
        if abs(np.max(np.abs(fxn))) < epsilon:
            return xn
        Dfxn = Df(xn)
        if np.max(np.abs(Dfxn)) < epsilon:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - np.linalg.inv(Dfxn)@fxn
    print('Exceeded maximum iterations. No solution found.')
    return None

def f(var,X,Y):
    u,v = var
    return np.array([33*np.cos(u+v)+30*np.cos(u)-X,33*np.sin(u+v)+30*np.sin(u)-Y])

def df(var):
    u,v = var
    return np.array([[-33*np.sin(u+v)-30*np.sin(u),-33*np.sin(u+v)],[33*np.cos(u+v)+30*np.cos(u),33*np.cos(u+v)]])

def newton_gravity(fg,Dfg,epsilon,max_iter,X,Y,x0 = np.array([0.8,1.5]),alpha = 0):
    x0 += alpha
    xn = x0
    for n in range(0,max_iter):
        fxn = fg(xn,X,Y,alpha)
        if abs(np.max(np.abs(fxn))) < epsilon:
            u,v = xn
            print(33*np.cos(u+v+alpha)+30*np.cos(u+alpha),33*np.sin(u+v+alpha)+30*np.sin(u+alpha),X,Y,u,v)
            xn[0] = np.abs(xn[0])%(2*np.pi)
            xn[1] = np.abs(xn[1])%(2*np.pi)
            u,v = xn
            print(33*np.cos(u+v+alpha)+30*np.cos(u+alpha),33*np.sin(u+v+alpha)+30*np.sin(u+alpha),X,Y,u,v)
            return xn
        Dfxn = Dfg(xn,alpha)
        if np.max(np.abs(Dfxn)) < epsilon:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - np.linalg.inv(Dfxn)@fxn
    print('Exceeded maximum iterations. No solution found.')
    u,v = xn
    print(33*np.cos(u+v+alpha)+30*np.cos(u+alpha),33*np.sin(u+v+alpha)+30*np.sin(u+alpha),X,Y)
    return None

def fg(var,X,Y,alpha):
    u,v = var
    return np.array([33*np.cos(u+v+alpha)+30*np.cos(u+alpha)-X,33*np.sin(u+v+alpha)+30*np.sin(u+alpha)-Y])

def dfg(var,alpha):
    u,v = var
    return np.array([[-33*np.sin(u+v+alpha)-30*np.sin(u+alpha),-33*np.sin(u+v+alpha)],[33*np.cos(u+v+alpha)+30*np.cos(u+alpha),33*np.cos(u+v+alpha)]])