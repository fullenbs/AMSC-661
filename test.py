import meshio
import gmsh
import pygmsh
from distmesh import get_verts as get_verts
import matplotlib.pyplot as plt
import numpy as np 

#Reads in a .msh file and plots it
def plot_msh(filename, oname):
    mesh = meshio.read(filename)
    pts = mesh.points[:, :2]

    triangles = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            break

    fig, ax = plt.subplots()
    ax.triplot(pts[:, 0], pts[:, 1], triangles, color='blue', lw=0.5)

    ax.set_aspect('equal')
    ax.set_title(f"Mesh: {filename}")
    plt.savefig(oname + '.png')


#Does the square with upper right corner removed
def square():
    with pygmsh.occ.Geometry() as geom:
        r1 = geom.add_rectangle([-1, -1, 0], 2, 2, mesh_size=0.05)
        r2 = geom.add_rectangle([0, 0, 0], 1, 1, mesh_size=0.05)
        fin = geom.boolean_difference(r1, r2)
        mesh = geom.generate_mesh()
        pygmsh.write("test.msh")
    plot_msh('test.msh', 'package_1')

#Pentagon with inner pentagon turned upside down removed
def pentagon(): 
    #Helper function I put in distmesh.py(), gets the vertices
    X = get_verts(1, 5)
    Y = get_verts(0.5, 5, rot=np.pi)

    with pygmsh.occ.Geometry() as geom:
        outer = geom.add_polygon(X,mesh_size=0.1)
        inner = geom.add_polygon(Y, mesh_size=0.1)
        fin = geom.boolean_difference(outer, inner)
        mesh = geom.generate_mesh()
        pygmsh.write("test2.msh")
    plot_msh('test2.msh', 'package_2')

#semi_ellipse with two circles removed
def semi_ellipse():
    with pygmsh.occ.Geometry() as geom:
        disk = geom.add_disk([0, 0], 2, 1, mesh_size=0.01)
        rect = geom.add_rectangle([-2, 0, 0], 4, 1, mesh_size=0.01)
        circle1 = geom.add_disk([-0.8, -0.4, 0], 0.2)
        circle2 = geom.add_disk([0.8, -0.4, 0], 0.2)
        fin = geom.boolean_difference(disk, [rect, circle1, circle2])
        mesh = geom.generate_mesh(dim=2)
        pygmsh.write("test3.msh")
    plot_msh('test3.msh', 'package_3')


if __name__ == '__main__': 
    #Just call function here
    semi_ellipse()