import numpy as np
import matplotlib.pyplot as plt

#Lax friedrichs 
def lf(u_initial, x, ts, nx=0, dx=0, nt=0, dt=0): 
    ut = np.zeros((x.shape[0], len(ts)))
    u = u_initial.copy()
    c = 0
    for n in range(nt):
        u_old = u.copy()
        
        f_right = 0.5*u_old[2:]**2
        f_left  = 0.5*u_old[:-2]**2
        
        u[1:-1] = 0.5*(u_old[2:] + u_old[:-2]) - \
                (dt/(2*dx))*(f_right - f_left)
        
        
        u[0] = u[1]
        u[-1] = u[-2]
        if n*dt in ts: 
            ut[:,c] = u.copy()
            c += 1
    return (u, ut)

#Mccormack's method
def mccormack(u_initial, x, ts, nx=0, dx=0, nt=0, dt=0):
    ut = np.zeros((x.shape[0], len(ts)))
    u = u_initial.copy()
    c = 0
    for n in range(nt):
        u_old = u.copy()
        f = 0.5 * u_old**2
        
        u_star = u_old.copy()
        u_star[:-1] = u_old[:-1] - (dt/dx) * (f[1:] - f[:-1])

        f_star = 0.5*u_star**2
        
        u[1:] = 0.5*(u_old[1:] + u_star[1:] - (dt/dx)*(f_star[1:] - f_star[:-1]))
        u[0], u[-1] = u[1], u[-2]

        if n*dt in ts: 
            ut[:,c] = u.copy()
            c += 1
    return (u, ut)

def richtmeyer(u_initial, x, ts, nx=0, dx=0, nt=0, dt=0): 
    ut = np.zeros((x.shape[0], len(ts)))
    u = u_initial.copy()
    c = 0
    for n in range(nt):
        u_old = u.copy()
        f_old = 0.5*u_old**2

            
        u_half = 0.5*(u_old[1:] + u_old[:-1]) - \
                (dt/(2*dx))*(f_old[1:] - f_old[:-1])
        
        f_half = 0.5*u_half**2
        print(f_old.shape, f_half.shape, u_half.shape)
        u[1:-1] = u_old[1:-1] - (dt/dx) * (f_half[1:] - f_half[:-1])
        
        u[0], u[-1] = u[1], u[-2]
        if n*dt in ts: 
            ut[:,c] = u.copy()
            c += 1
    return (u, ut)

def godunov(u_initial, x, ts, nx=0, dx=0, nt=0, dt=0): 
    ut = np.zeros((x.shape[0], len(ts)))
    u = u_initial.copy()
    c = 0
    for n in range(nt):
        u_old = u.copy()
        f_old = 0.5*u_old**2
        u[1:-1] = u_old[1:-1] - (dt/dx)*(f_old[1:-1] - f_old[0:-2])
        u[0], u[-1] = u[1], u[-2]
        if n*dt in ts: 
            ut[:,c] = u.copy()
            c += 1
    return (u, ut)


#Exact solution to Burgers equation w/: 
#u_0(x) = 0 x < 0, 2 on 0 < x < 1, 1 on 1 < x <2 and o otherwise
def exact_sol_1(x, ts): 
    u_sol = np.zeros((x.shape[0], len(ts)))
    for i in range(len(ts)): 
        t = ts[i]
        if t <= 1: 
            for j in range(0, len(x)): 
                if x[j] > 0 and x[j] < 2*t: 
                    u_sol[j, i] = x[j]/t 
                elif x[j] > 2*t and x[j] <= 1.5*t + 1: 
                    u_sol[j, i] = 2 
                elif x[j] > 1.5*t + 1 and x[j] <= t/2 + 2: 
                    u_sol[j, i] = 1
                else: 
                    u_sol[j, i] = 0
        elif t > 1 and t <= 1.5: 
            for j in range(0, len(x)): 
                if x[j] > 0 and x[j] < 2*t: 
                    u_sol[j, i] = x[j]/t 
                elif x[j] > 2*t and x[j] <= 1.5 + t: 
                    u_sol[j, i] = 2 
                else: 
                    u_sol[j, i] = 0
        elif t > 1.5: 
            for j in range(0, len(x)): 
                if x[j] > 0 and x[j] < np.sqrt(6*t): 
                    u_sol[j, i] = x[j]/t 
                else: 
                    u_sol[j, i] = 0
    return u_sol


#Exact solution to Burgers' IVP where u_0(x) = 1 for x on (0, 1), 0 otherwise
def exact_sol_2(x, ts): 
    u_sol = np.zeros((x.shape[0], len(ts)))
    for i in range(len(ts)): 
        t = ts[i]
        if t <= 2: 
            for j in range(0, x.shape[0]): 
                if 0 < x[j] and x[j] < t: 
                    u_sol[j, i] = x[j]/t 
                elif x[j] > t and x[j] < 1 + t/2: 
                    u_sol[j, i] = 1 
                else: 
                    u_sol[j, i] = 0
        else: 
            for j in range(0, x.shape[0]): 
                if 0 < x[j] and x[j] < np.sqrt(2*t): 
                    u_sol[j, i] = x[j]/t 
                else: 
                    u_sol[j, i] = 0
    return u_sol

def exact_sol_plot(p, x, u_initial): 
    ts = []
    if p == 1: 
        ts = [0.5, 1.5, 2.5, 3.5, 5]
        ut = exact_sol_1(x, ts)
    elif p == 2: 
        ts = [0, 1, 2, 3, 4, 5, 6]
        ut = exact_sol_2(x, ts)
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_initial, '--', label="Initial Condition", alpha=0.5)
    plt.title("Exact solution to Burgers' Equation IVP")
    for n in range(len(ts)): 
        plt.plot(x, ut[:,n], label=f'State (t={ts[n]:.2f})', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(f'exact_sol_{p}')
    

def problem2(L, x, nx, dx, dt, nt, ts, u_initial, p, func=lf): 
    (u, ut) = func(u_initial, x, ts, nx, dx, nt, dt)
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_initial, '--', label="Initial Condition", alpha=0.5)
    plt.title("Burgers' Equation w/ MacCormacks:")
    for n in range(len(ts)): 
        plt.plot(x, ut[:,n], label=f'State (t={ts[n]:.2f})', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(f'mcp{p}')

if __name__ == '__main__': 
    L = 9
    nx = 300   
    dx = L/(nx - 1)
    dt = 0.005
    nt = 1000 
    #(dx, dt) = (0.01, 0.01)
    print(dt, dx)
    exit()

    x = np.linspace(-2, L - 1, nx)
    u_initial = np.zeros(nx)
    # u_initial[(x >= 0) & (x <= 1)] = 2.0
    # u_initial[(x > 1) & (x <= 2)] = 1.0
    u_initial[(x >= 0) & (x <= 1)] = 1.0

    ts = [0, 1, 2, 3, 4, 5, 6]
    problem2(L, x, nx, dx, dt, nt, ts, u_initial, 4, func=lf)

