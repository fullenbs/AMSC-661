import numpy as np
import matplotlib.pyplot as plt
import pygmsh
import meshio
import numpy as np
from scipy.interpolate import CubicSpline

#Global r1/r2/m parameters for the star
r1 = 2
r2 = 0.5
m_star = 5


#Reads in the mesh
def load_msh(filename):
    mesh = meshio.read(filename)
    pts = mesh.points[:, :2]

    triangles = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            break    
    return pts, triangles 

#Dr.Cameron's curve related functions: 
# Define the curve
def star_curve_rad(t):
    return r1 + r2 * np.cos(m_star*t)

def star_curve_drad(t):
    return -r2*m_star*np.sin(m_star*t)
    
def star_curve_speed(t):
    r = star_curve_rad(t)
    dr = star_curve_drad(t)
    return np.sqrt(r**2 + dr**2)

def star_curve_normal(t):
    r = star_curve_rad(t)
    dr = star_curve_drad(t)
    speed = np.sqrt(r**2 + dr**2)
    v1 = np.array([np.cos(t),np.sin(t)])
    v2 = np.array([np.sin(t),-np.cos(t)])
    normal = (r*v1 + dr*v2)/speed
    return normal

#Gets point from angle and vice versa
def get_point(t):
    r = star_curve_rad(t)
    return np.array([r*np.cos(t),r*np.sin(t)])

#Exact solution for interior problem
def exact_sol1(x):
    r = np.linalg.norm(x)
    phi = np.arctan2(x[1],x[0]) 
    return 1 + r**3*np.cos(3*phi)
    
#Boundary of interior problem
def bdry_func(t):
    r = star_curve_rad(t)
    return 1 + (r**3)*np.cos(3*t)

#My curve related functions, these are for exterior problem:
def ext_exact2(x): 
    r = np.linalg.norm(x)
    phi = np.arctan2(x[1], x[0])
    return ext_exact(r, phi)

#Exterior exact given (r, phi)
def ext_exact(r, t): 
    return r**(-3)*np.cos(3*t) + np.log(r)

#Boundary for exterior problem
def ext_bdry(t): 
    return (r1 + r2*np.cos(m_star*t))**(-3)*np.cos(3*t) + np.log(r1 + r2*np.cos(m_star*t))

#Generates the mesh: 
#mode='ext' creates the exterior mesh so rectangle with a star cut out
#mode='int/anything else' creates the interior mesh, so just the star
#Saves the result to a file which I then just load in from in the future
#to not have to recreate it every time.
def mesh_gen(mode='ext'): 
    h_mesh = 0.1
    N = 200
    t_vals = np.linspace(0, 2*np.pi, N, endpoint=False)

    points = np.array([
        [star_curve_rad(t)*np.cos(t), star_curve_rad(t)*np.sin(t), 0.0]
        for t in t_vals
    ])

    if mode != 'ext': 
        with pygmsh.geo.Geometry() as geom:
            # Add points
            pts = [geom.add_point(p, mesh_size=h_mesh) for p in points]
            pts.append(pts[0])

            curve = geom.add_spline(pts)
            loop = geom.add_curve_loop([curve])
            surface = geom.add_plane_surface(loop)

            mesh = geom.generate_mesh()
            pygmsh.write("interior.msh")
    else: 
        with pygmsh.occ.Geometry() as geom: 

            #Adds points
            pts = [geom.add_point(p, mesh_size=h_mesh) for p in points]
            pts.append(pts[0])
            spline = geom.add_spline(pts)
            loop = geom.add_curve_loop([spline])
            surface = geom.add_plane_surface(loop)
            rect = geom.add_rectangle([-4, -4, 0], 8, 8)

            geom.boolean_difference(rect, surface)

            mesh = geom.generate_mesh()
            pygmsh.write("exterior.msh")

    tri = mesh.cells_dict["triangle"]
    pts = mesh.points[:, :2]  # ignore z
    return pts, tri


#Finds boundary nodes and interior points
def bdry_int(pts, tri, plot): 
    Ntri = np.size(tri,axis = 0)

    Nedges = Ntri*3
    edges = np.zeros((Nedges,2),dtype = int)
    for j in range(Ntri):
        t = tri[j,:]
        edges[3*j,:] = np.sort(np.array([t[0],t[1]]))
        edges[3*j+1,:] = np.sort(np.array([t[0],t[2]]))
        edges[3*j+2,:] = np.sort(np.array([t[1],t[2]]))

    # Indices of boundary points
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    ind = np.argwhere(counts == 1)
    Bdry = np.unique(unique_edges[ind,:].ravel())

    # Indices of interior points
    # Note that pts contains not only mesh points 
    # but also points on the boundary curve that did not become part of the mesh
    mesh_pts = np.unique(unique_edges.ravel())
    Interior = np.setdiff1d(mesh_pts,Bdry)

    if plot: 
        plt.figure()
        plt.triplot(pts[:,0], pts[:,1], tri,linewidth = 0.5)
        plt.scatter(pts[Bdry,0],pts[Bdry,1],s = 2, c = 'red')
        plt.scatter(pts[Interior,0],pts[Interior,1],s = 1, c = 'black')
        plt.gca().set_aspect('equal')
    return Bdry, Interior

#Calculates D for interior problem
def Dmatrix_trapezoid(p):
    Nc = p.shape[0]
    D = np.zeros((Nc, Nc))
    
    #Determines weights for composite trapezoid rule
    dt = 2*np.pi/Nc
    for i in range(Nc):
        x = get_point(p[i])
        for j in range(Nc):
            if i != j:
                y = get_point(p[j])
                n_j = star_curve_normal(p[j])
                speed_j = star_curve_speed(p[j])
                D[i, j] = star_kernel(x, y, n_j) * speed_j * dt
            else:
                p1 = p[j] + 0.1*dt
                p2 = p[j] - 0.1*dt
                y1 = get_point(p1)
                s1 = star_curve_speed(p1)
                s2 = star_curve_speed(p2)
                y2 = get_point(p2)
                D[i, j] = 0.5*(star_kernel(x, y1, star_curve_normal(p1))*s1 + star_kernel(x, y2, star_curve_normal(p2))*s2)*dt
    return D 

#Kernel for interior problem
def star_kernel(x, y, n_y):
    r_vec = x - y
    dist_sq = np.sum(r_vec**2)
    return (0.5/np.pi) * np.dot(r_vec, n_y) / dist_sq

#Distance from a point to a set, though really just use this for the boundary
def true_distance_to_boundary(x, bdry_pts):
    dists = np.linalg.norm(bdry_pts - x, axis=1)
    return np.min(dists)

#Main control loop for interior problem
def interior_problem():
    #To save time in the testing process
    pts, tri = load_msh('test.msh')
    Bdry, Interior = bdry_int(pts, tri, plot=False)

    Nc = 200
    p = np.linspace(0, 2*np.pi, Nc + 1)
    p = np.delete(p, -1)

    #Solves for sigma
    D = Dmatrix_trapezoid(p)
    A = -0.5*np.eye(Nc) + D 
    f = np.array([bdry_func(t) for t in p])
    sigma = np.linalg.solve(A, f)

    #Now gets u 
    u = np.zeros(len(pts))

    #Updates interior
    Dvec = np.zeros((Nc,))
    for j in range(len(Interior)):
        ind = Interior[j]
        x = pts[ind]
        for k in range(Nc): 
            Dvec[k] = star_kernel(x, get_point(p[k]), star_curve_normal(p[k]))*star_curve_speed(p[k])
        u[ind] = np.dot(Dvec, sigma)*2*np.pi/Nc
    
    #Updates boundary
    Nb = np.size(Bdry)
    for j in range(Nb):
        ind = Bdry[j]
        x = pts[ind]
        u[ind] = exact_sol1(x)

    #Error calc and related
    u_exact = np.array([exact_sol1(p) for p in pts])
    solved_indices = np.concatenate([Bdry, Interior]).astype(int)
    err_calc = u[solved_indices] - u_exact[solved_indices]

    rmse = np.sqrt(np.sum(err_calc**2)/pts.shape[0])
    max_err = np.max(np.abs(err_calc))
    print(f"RMSE = {rmse:.4e}, max_err = {max_err:.4e}")

    #Spline setup
    p_aux = np.zeros((Nc+1,))
    sigma_aux = np.zeros((Nc+1,))
    p_aux[:Nc] = p
    p_aux[-1] = 2*np.pi
    sigma_aux[:Nc] = sigma
    sigma_aux[-1] = sigma[0]

    cs = CubicSpline(p_aux,sigma_aux, bc_type='periodic')

    p_fine = np.linspace(0, 2*np.pi, 50*Nc+1)
    sigma_fine = cs(p_fine)

    p_fine = np.delete(p_fine,-1)
    sigma_fine = np.delete(sigma_fine,-1)
    Nc_fine = np.size(p_fine)
    y_fine = np.array([get_point(p) for p in p_fine])
    speed_fine = np.array([star_curve_speed(t) for t in p_fine])
    n_j_fine = np.array([star_curve_normal(t) for t in p_fine])
 
    # Interior points update
    dt = 2*np.pi/Nc_fine
    Ni = np.size(Interior)
    Bdry_set = pts[Bdry]
    for j in range(Ni):
        ind = Interior[j]
        x = pts[ind,:]
        Dvec_fine = np.zeros((Nc_fine,))
        if true_distance_to_boundary(x, Bdry_set) <= 0.2:
            for k in range(Nc_fine):
                Dvec_fine[k] = star_kernel(x, y_fine[k], n_j_fine[k])*speed_fine[k]  
            u[ind] = np.dot(Dvec_fine, sigma_fine)*dt

    err = u - u_exact
    err_calc = u[solved_indices] - u_exact[solved_indices]

    rmse = np.sqrt(np.sum(err_calc**2)/pts.shape[0])
    max_err = np.max(np.abs(err_calc))
    print(f"RMSE = {rmse:.4e}, max_err = {max_err:.4e}")

    plt.figure()
    plt.tricontourf(pts[:,0],pts[:,1], tri, err, levels=200, cmap='turbo')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title("Error plot")

    plt.figure()
    plt.tricontourf(pts[:,0],pts[:,1], tri, u, levels=200, cmap='turbo')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title("Calculated solution")

    plt.figure()
    plt.tricontourf(pts[:,0],pts[:,1], tri, u_exact, levels=200, cmap='turbo')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title("Exact solution")
    plt.show()


#Calculates D for exterior problem
def Dmatrix_exterior(p):
    Nc = p.shape[0]
    D = np.zeros((Nc, Nc))
    
    dt = 2*np.pi/Nc
    for i in range(Nc):
        x = get_point(p[i])
        for j in range(Nc):
            if i != j:
                y = get_point(p[j])
                n_j = star_curve_normal(p[j])
                speed_j = star_curve_speed(p[j])
                D[i, j] = star_kernel2(x, y, n_j) * speed_j * dt
            else:
                p1 = p[j] + 0.1*dt
                p2 = p[j] - 0.1*dt
                y1 = get_point(p1)
                s1 = star_curve_speed(p1)
                s2 = star_curve_speed(p2)
                y2 = get_point(p2)
                D[i, j] = 0.5*(star_kernel2(x, y1, star_curve_normal(p1))*s1 + star_kernel2(x, y2, star_curve_normal(p2))*s2)*dt
    return D 

#Kernel for exterior problem
def star_kernel2(x, y, n_y):
    r_vec = x - y
    dist_sq = np.sum(r_vec**2)
    return (0.5/np.pi) * np.dot(r_vec, n_y) / dist_sq - (0.5/np.pi)*np.log(np.linalg.norm(x))

#Control loop for exterior problem, almost about the same as for interior problem
def exterior_problem():
    pts, tri = load_msh('exterior.msh')
    Bdry, Interior = bdry_int(pts, tri, plot=False)

    Nc = 200
    p = np.linspace(0, 2*np.pi, Nc + 1)
    p = np.delete(p, -1)

    #Solves for sigma
    D = Dmatrix_exterior(p)

    A = 0.5*np.eye(Nc) + D 
    f = np.array([ext_bdry(t) for t in p])
    sigma = np.linalg.solve(A, f)

    #Now gets u 
    u = np.zeros(len(pts))

    Dvec = np.zeros((Nc,))
    for j in range(len(Interior)):
        ind = Interior[j]
        x = pts[ind]
        for k in range(Nc): 
            Dvec[k] = star_kernel2(x, get_point(p[k]), star_curve_normal(p[k]))*star_curve_speed(p[k])
        u[ind] = np.dot(Dvec, sigma)*2*np.pi/Nc 

    #Bdry
    Nb = np.size(Bdry)
    for j in range(Nb):
        ind = Bdry[j]
        x = pts[ind]
        u[ind] = ext_exact2(x)

    u_exact = np.array([ext_exact2(p) for p in pts])
    solved_indices = np.concatenate([Bdry, Interior]).astype(int)
    err_calc = u[solved_indices] - u_exact[solved_indices]

    rmse = np.sqrt(np.sum(err_calc**2)/pts.shape[0])
    max_err = np.max(np.abs(err_calc))
    print(f"RMSE = {rmse:.4e}, max_err = {max_err:.4e}")

    p_aux = np.zeros((Nc+1,))
    sigma_aux = np.zeros((Nc+1,))
    p_aux[:Nc] = p
    p_aux[-1] = 2*np.pi
    sigma_aux[:Nc] = sigma
    sigma_aux[-1] = sigma[0]

    cs = CubicSpline(p_aux,sigma_aux, bc_type='periodic')

    p_fine = np.linspace(0, 2*np.pi, 20*Nc+1)
    sigma_fine = cs(p_fine)

    p_fine = np.delete(p_fine,-1)
    sigma_fine = np.delete(sigma_fine,-1)
    Nc_fine = np.size(p_fine)
    y_fine = np.array([get_point(p) for p in p_fine])
    speed_fine = np.array([star_curve_speed(t) for t in p_fine])
    n_j_fine = np.array([star_curve_normal(t) for t in p_fine])
 
    # Interior mesh points
    dt = 2*np.pi/Nc_fine
    Ni = np.size(Interior)
    Bdry_set = pts[Bdry]
    for j in range(Ni):
        ind = Interior[j]
        x = pts[ind,:]
        Dvec_fine = np.zeros((Nc_fine,))
        if true_distance_to_boundary(x, Bdry_set) <= 0.4:
            for k in range(Nc_fine):
                Dvec_fine[k] = star_kernel2(x, y_fine[k], n_j_fine[k])*speed_fine[k]  
            u[ind] = np.dot(Dvec_fine, sigma_fine)*dt

    err = u - u_exact
    err_calc = u[solved_indices] - u_exact[solved_indices]

    rmse = np.sqrt(np.sum(err_calc**2)/pts.shape[0])
    max_err = np.max(np.abs(err_calc))
    print(f"RMSE = {rmse:.4e}, max_err = {max_err:.4e}")

    #Plotting 
    plt.figure()
    plt.tricontourf(pts[:,0],pts[:,1], tri, err, levels=200, cmap='turbo')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title('Error plot')
    plt.savefig("Error plot")

    plt.figure()
    plt.tricontourf(pts[:,0],pts[:,1], tri, u, levels=200, cmap='turbo')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title('Calculated solution')
    plt.savefig("Calculated solution")

    plt.figure()
    plt.tricontourf(pts[:,0],pts[:,1], tri, u_exact, levels=200, cmap='turbo')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title("Exact solution")
    plt.savefig('Exact solution')


if __name__ == '__main__': 
   exterior_problem()