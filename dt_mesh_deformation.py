from mesh import Mesh
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# Currently only 2D support

class DT_Mesh_Deform:
    @staticmethod
    def visualize_delaunay(vertices):
        vertices = np.array(vertices)
        tri = Delaunay(vertices)
        plt.triplot(vertices[:,0], vertices[:,1], tri.simplices)
        plt.plot(vertices[:,0], vertices[:,1], 'o')
        plt.show()

    @staticmethod
    def deform(mesh, exterior, exterior_mapping):
        tri = Delaunay(exterior)
        simplices = tri.simplices

        v_tris = tri.find_simplex(mesh.vertices)

        new_vertices = []
        for i in range(len(mesh.vertices)):
            x = mesh.vertices[i][0]
            y = mesh.vertices[i][1]

            triangle = simplices[v_tris[i]]
            triangle = [(exterior[triangle[0]][0], exterior[triangle[0]][1]), (exterior[triangle[1]][0], exterior[triangle[1]][1]), (exterior[triangle[2]][0], exterior[triangle[2]][1])]


            A = DT_Mesh_Deform.triangle_area(triangle[0], triangle[1], triangle[2])

            e1 = DT_Mesh_Deform.triangle_area(triangle[0], triangle[1], (x,y)) / A
            e2 = DT_Mesh_Deform.triangle_area(triangle[1], triangle[2], (x,y)) / A
            e3 = DT_Mesh_Deform.triangle_area(triangle[0], triangle[2], (x,y)) / A

            triangle = [exterior_mapping[t] for t in triangle]

            tx = e1 * triangle[2][0] + e2 * triangle[0][0] + e3 * triangle[1][0]
            ty = e1 * triangle[2][1] + e2 * triangle[0][1] + e3 * triangle[1][1]

            new_vertices.append((tx,ty))

        return Mesh(new_vertices, mesh.faces, mesh.dim)

    @staticmethod
    def triangle_area(p1, p2, p3):
        mat = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
        return abs(0.5 * np.linalg.det(mat))


