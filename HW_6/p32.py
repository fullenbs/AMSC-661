from distmesh import *
import numpy as np
import matplotlib.pyplot as plt
import scipy

def mesh_gen():
    n = 60
    t = np.linspace(0,2*np.pi*(n/(n+1)),n)
    rad = 1.0
    h0 = 0.1# the desired side of mesh triangles

    pfix = np.concatenate((np.reshape(rad*np.cos(t),(n,1)),np.reshape(rad*np.sin(t),(n,1))),axis = 1)
    pfix[:,0] = pfix[:,0] + 1.5 
    pfix[:,1] = pfix[:,1] + 1.5
    # indexes of angle values for defining ears
    f = lambda p: drectangle(p, 0, 3, 0, 3)
    fh = huniform

    bbox = [0,3,0,3] # the bounding box

    pts,tri = distmesh2D(f,fh,h0,bbox,pfix)
    return pts, tri, pfix

def stima3(verts):
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    rhs = np.zeros((3,2))
    rhs[1,0] = 1
    rhs[2,1] = 1
    G = np.zeros((3,2))
    G[:,0] = np.linalg.solve(Aux,rhs[:,0])
    G[:,1] = np.linalg.solve(Aux,rhs[:,1])
    M = 0.5*np.linalg.det(Aux)*np.matmul(G,np.transpose(G))
    return M

def FEM(pts, tri, d_bdry, idx_x0, idx_x1, a1=1.2, a2=1):
    Npts = np.size(pts,axis=0) # the number of mesh points
    Ntri = np.size(tri,axis=0) # the number of triangle
    print("Npts = ",Npts," Ntri = ",Ntri)

    free_nodes = np.setdiff1d(np.arange(0,Npts,1,dtype = int),d_bdry,assume_unique=True)

    A = scipy.sparse.csr_matrix((Npts,Npts), dtype = float).toarray() # define the sparse matrix A
    b = np.zeros((Npts,1)) # the right-hand side
    u = np.zeros((Npts,1)) # the solution
    u[idx_x1] = 1 # define u at known values
    u[idx_x0] = 0

    # stiffness matrix
    a_c = 1
    for j in range(Ntri):
        v = pts[tri[j,:],:] # vertices of mesh triangle
        centroid = np.mean(v, axis=0)
        if (centroid[0] - 1.5)**2 + (centroid[1] - 1.5)**2 < 1: 
            a_c = a1
        else: 
            a_c = a2
        ind = tri[j,:]
        indt = np.array(ind)[:,None]
        A[np.ix_(ind, ind)] += a_c * stima3(v)
        #A[indt,ind] = A[indt,ind] + a_c*stima3(v)

    # load vector
    b = b - np.matmul(A,u)

    free_nodes_t = np.array(free_nodes)[:,None]
    u[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    u = np.reshape(u,(Npts,))
    return u

#Gets the gradient
def compute_element_gradients(pts, tri, u, a1=1.2, a2=1.0):
    num_tri = len(tri)
    grad_x_elements = np.zeros(num_tri)
    grad_y_elements = np.zeros(num_tri)

    a_c = 1
    for i, tr in enumerate(tri):
        coords = pts[tr]
        u_tri = u[tr] 
        centroid = np.mean(coords, axis=0)
        if (centroid[0] - 1.5)**2 + (centroid[1] - 1.5)**2 < 1: 
            a_c = a1
        else: 
            a_c = a2

        # Compute gradient
        M = np.column_stack([np.ones(3), coords])
        grad = A @ np.linalg.solve(M, u_tri)

        grad_x_elements[i] = -1*a_c*grad[0]
        grad_y_elements[i] = -1*a_c*grad[1]
        
    return grad_x_elements, grad_y_elements

def get_node_grad(pts, tri, grad_x_e, grad_y_e):
    num_pts = len(pts)
    grad_x_nodes = np.zeros(num_pts)
    grad_y_nodes = np.zeros(num_pts)
    count = np.zeros(num_pts)

    for i, tr in enumerate(tri):
        # Add the triangle's constant gradient to each of its 3 nodes
        for node_idx in tr:
            grad_x_nodes[node_idx] += grad_x_e[i]
            grad_y_nodes[node_idx] += grad_y_e[i]
            count[node_idx] += 1

    safe_count = np.where(count > 0, count, 1)
    return grad_x_nodes/safe_count, grad_y_nodes/safe_count


if __name__ == '__main__':
    pts, tri, pfix = mesh_gen()
    for pt in pts: 
        if pt[0] > 3 or pt[0] < 0: 
            print(pt) 

    idx_x0 = np.where(np.isclose(pts[:, 0], 0, atol=1e-6))[0]
    idx_x1 = np.where(np.isclose(pts[:, 0], 3, atol=1e-6))[0]

    d_bdry = np.concatenate((idx_x0, idx_x1))
    # plt.scatter(pts[:,0], pts[:,1], color='blue') 
    # plt.scatter(pfix[:,0], pfix[:,1], color='green')
    # plt.scatter(pts[d_bdry, 0], pts[d_bdry,1], color='red')
    # plt.show()
    # exit()

    u = FEM(pts, tri, d_bdry, idx_x0, idx_x1, a1=1.2, a2=1)
    # gx1, gy1 = compute_element_gradients(pts, tri, u, a1=1, a2=1)
    # gx, gy = get_node_grad(pts, tri, gx1, gy1)

    # Plot the magnitude as a contour
    # mag = np.sqrt(gx**2 + gy**2)
    # plt.tricontourf(pts[:,0], pts[:,1], tri, mag, np.arange(0, 1.1, 0.1), cmap='magma')


    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize = (8,8))
    plt.tricontourf(pts[:,0], pts[:,1],tri,u,np.arange(0,1.1,0.1), cmap='magma')
    #plt.scatter(pts[:,0], pts[:,1])
    plt.colorbar()
    plt.show()
