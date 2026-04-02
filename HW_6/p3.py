import distmesh as distmesh
import numpy as np 
import matplotlib.pyplot as plt
import scipy 

def mesh_gen2():
    n = 100
    t = np.linspace(0,2*np.pi*(n/(n+1)),n)
    rad = 1.0
    h0 = rad*2*np.pi/n # the desired side of mesh triangles

    pfix = np.concatenate((np.reshape(rad*np.cos(t),(n,1)),np.reshape(rad*np.sin(t),(n,1))),axis = 1)
    pfix[:,0] = pfix[:,0] + 1.5 
    pfix[:,1] = pfix[:,1] + 1.5
    # indexes of angle values for defining ears
    f = lambda p: distmesh.drectangle(p, 0, 3, 0, 3)
    fh = distmesh.huniform

    bbox = [0,3,0,3] # the bounding box

    pts,tri = distmesh.distmesh2D(f,fh,h0,bbox,pfix)
    return pts, tri


def assembly(pts, triangles, f_source=0, a1=1.2, a2=1):
    num_pts = len(pts)
    K_global = np.zeros((num_pts, num_pts))
    F = np.zeros(num_pts)

    a_c = 1
    for tri in triangles:
        #Gets the coordinates and then the A_{ij}
        coords = pts[tri]
        centroid = np.mean(coords, axis=0)
        if (centroid[0] - 1.5)**2 + (centroid[1] - 1.5)**2 < 1: 
            a_c = a1
        else: 
            a_c = a2
            
        M = np.column_stack([np.ones(3), coords])
        A = 0.5 * np.abs(np.linalg.det(M))

        inv_M = np.linalg.inv(M)
        grad_N = inv_M[:, 1:]  # (3, 2)

        ke = A * a_c * (grad_N @ grad_N.T)

        #Gets right index and adds accordingly
        K_global[np.ix_(tri, tri)] += ke

        fe = (f_source*A/3.0) * np.ones(3)
        F[tri] += fe

    return (K_global, F)

def enforce_bdry(K, F, pts):
    idx_x0 = np.where(np.isclose(pts[:, 0], 0.0, atol=1e-5))[0]
    idx_x1 = np.where(np.isclose(pts[:, 0], 3.0, atol=1e-5))[0]
    print(idx_x0, idx_x1)

    # Enforce u = 0 at x = 0
    for idx in idx_x0:
        K[idx, :] = 0      # Clear the row
        K[idx, idx] = 1    # Put 1 on the diagonal
        F[idx] = 0.0       # Set target value
        
    # Enforce u = 1 at x = 3
    for idx in idx_x1:
        K[idx, :] = 0      # Clear the row
        K[idx, idx] = 1    # Put 1 on the diagonal
        F[idx] = 1.0       # Set target value
    
    return K, F

if __name__ == '__main__': 
    pts, tri = mesh_gen2()
    (K, F) = assembly(pts, tri)
    (K, F) = enforce_bdry(K, F, pts)
    u = np.linalg.solve(K, F)
    Npts = len(pts) 
    plt.figure(figsize=(10, 8))
    contour = plt.tricontourf(pts[:, 0], pts[:, 1], tri, u, levels=20, cmap='viridis')
    
    plt.colorbar(contour, label='Solution Value (u)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('FEM Solution u(x,y)')
    
    # Optional: Overlay the mesh edges to see the triangles
    plt.triplot(pts[:, 0], pts[:, 1], tri, color='white', lw=0.5, alpha=0.3)
    
    plt.axis('equal')
    plt.show()

    #Make circle and feed that in as pfix, not the circle removal stuff
