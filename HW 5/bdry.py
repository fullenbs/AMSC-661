import numpy as np 
import matplotlib.pyplot as plt 

#Since no specifications were provided
def solve(size):
    x_min, x_max = -np.pi, np.pi
    y_min, y_max = 0, 2
    
    hx = (x_max - x_min)/(size - 1)
    hy = (y_max - y_min)/(size - 1)
    
    n_int = size
    x_int = np.linspace(x_min + hx, x_max - hx, n_int)
    y_int = np.linspace(y_min + hy, y_max - hy, n_int)
 
    X, Y = np.meshgrid(x_int, y_int)

    T1 = np.diag(np.ones(n_int - 1), -1) + np.diag(np.ones(n_int - 1), 1)
    T1[0, n_int - 1] = 1
    T1[n_int - 1, 0] = 1
    T1 = T1/hx**2
    T = T1 + np.diag(np.ones(n_int) * (-2/hx**2 + -2/hy**2))

    I1 = (np.diag(np.ones(n_int-1), -1) + np.diag(np.ones(n_int-1), 1))
    I1[n_int -1, n_int - 2] = 2
    I1 = I1/hy**2

    I = np.eye(n_int)

    A = np.kron(I, T) + np.kron(I1, I)
    F = -1*np.cos(X)*((X >= -np.pi/2) & (X <= np.pi/2))

    b = F.flatten()
    sol = np.linalg.solve(A, b)
    sol = sol.reshape((n_int, n_int))

    plt.imshow(sol, origin='lower', cmap='magma')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Solution w/' + str(size) + 'x' + str(size) + ' grid')
    plt.savefig('Poisson_sol.png')

if __name__ == '__main__': 
    solve(100)

