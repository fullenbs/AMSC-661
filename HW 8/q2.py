import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Parameters
n = 100  
h = 2/(n + 1)           
t_span = (0, 1.2)
xi_interior = -1 + np.arange(1, n + 1)*h #Interior points
xi_full = np.linspace(-1, 1, n + 2) #Full domain

#Two different initial values
#u0 = np.maximum(0, 1 - np.abs(xi_interior)**2)
u0 = np.maximum(0, 1 - 0.99*np.cos(2*np.pi*xi_interior))

#Initial boundary
xL0 = -1
xR0 = 1
y0 = np.concatenate((u0, [xR0, xL0]))

#Gets ODE, does all the calculation
def boussineq(t, y, n, h):
    u = y[:n]
    xR = y[n]
    xL = y[n+1]
    
    #xf 
    xf = (xR - xL)/2
    
    #Approximations
    d_xi_u_L = (4*u[0] - u[1])/(2*h)
    d_xi_u_R = (-4*u[-1] + u[-2])/(2*h)
    
    #Derivatives wrt xf
    dxR_dt = -d_xi_u_R/xf
    dxL_dt = -d_xi_u_L/xf

    u_full = np.concatenate(([0], u, [0]))
    
    d_xi_u = (u_full[2:] - u_full[:-2])/(2*h)
    d2_xi_u = (u_full[2:] - 2*u_full[1:-1] + u_full[:-2])/(h**2)
    
    #Does it in terms of xi
    xi = -1 + np.arange(1, n + 1)*h
    bracket = (1 + xi)*d_xi_u_R + (1 - xi)*d_xi_u_L
    
    #Calculates derivative 
    dudt = (1/xf**2)*(
        -0.5*bracket*d_xi_u + 
        u*d2_xi_u + d_xi_u**2)
    return np.concatenate((dudt, [dxR_dt, dxL_dt]))

#Solves ivp
sol = solve_ivp(boussineq, t_span, y0, method='BDF', args=(n, h), 
                t_eval=np.linspace(0, 1.2, 13))


plt.figure(figsize=(10, 7))
cmap = plt.get_cmap('turbo')
colors = cmap(np.linspace(0, 1, len(sol.t)))

#This gets and plots the solutions at each time step in (\xi, x) time
#Didn't want to factorize/break it apart so just comment/uncomment depending 
#if want u(\xi, t) or u(x, t)
for i in range(len(sol.t)):
    #Pads solution with boundary
    u_t = np.pad(sol.y[:n, i], (1, 1), mode='constant')
    umax_t = np.max(u_t)
    xR_t = sol.y[n, i]
    xL_t = sol.y[n+1, i]
    
    x0_t = (xR_t + xL_t)/2
    xf_t = (xR_t - xL_t)/2
    print(x0_t, xf_t)
    
    #Converts to u(x, t) if needed
    #x_physical = x0_t + xi_full * xf_t
    x_physical = u_t/umax_t

    #Different plots
    #plt.plot(x_physical, u_t, color=colors[i], label=f't={sol.t[i]:.2f}')
    plt.plot(xi_full, x_physical, color=colors[i], label=f't={sol.t[i]:.2f}')

#Plots the curve, and then everything else
plt.plot(xi_full, 1 - xi_full**2, label=r'1 - \xi^2')
#plt.title(r'Solution with initial value $\max(0, 1 - 0.99\cos(2\pi x))$')

plt.title(r'Comparison plot for $\; u(\xi, t)/u_{max}(t)$ with initial value $\max(0, 1 - 0.99\cos(2\pi x)$')
#plt.title(r'Solution with initial value $\; max(0, 1 - 0.99\cos(2\pi x))$')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('u(x,t)fig22') #Just save figure
#plt.show()