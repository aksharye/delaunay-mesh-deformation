from mesh import Mesh
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from dt_mesh_deformation import DT_Mesh_Deform
from dtrbf_mesh_deformation import DTRBF1_Mesh_Deform

# Currently only 2D support
# RBF1 Method

class DTV_Mesh_Deform:
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
            
            dx = []
            dy = []
            for j in range(3):
                dx.append(abs(triangle[j][0] - exterior_mapping[triangle[j]][0]))
                dy.append(abs(triangle[j][1] - exterior_mapping[triangle[j]][1]))

            dx = np.array(dx)
            dy = np.array(dy)

            M = []
            for j in range(3):
                row = []
                for k in range(3):
                    row.append(DTV_Mesh_Deform.rbf(np.linalg.norm(np.array(triangle[j]) - np.array(triangle[k]))))
                M.append(row)

            M = np.array(M)
            if np.linalg.det(M) != 0:
                Minv = np.linalg.inv(M)
            else:
                new_vertices.append((x,y))
                continue

            alpha_x = np.dot(Minv, dx)
            alpha_y = np.dot(Minv, dy)

            A = []
            for j in range(3):
                A.append(DTV_Mesh_Deform.rbf(np.linalg.norm(np.array((x,y)) - np.array(triangle[j]))))
            A = np.array(A)

            dx = x + np.dot(A, alpha_x)
            dy = y + np.dot(A, alpha_y)

            
            A = DTV_Mesh_Deform.triangle_area(triangle[0], triangle[1], triangle[2])

            e1 = DTV_Mesh_Deform.triangle_area(triangle[0], triangle[1], (x,y)) / A
            e2 = DTV_Mesh_Deform.triangle_area(triangle[1], triangle[2], (x,y)) / A
            e3 = DTV_Mesh_Deform.triangle_area(triangle[0], triangle[2], (x,y)) / A

            triangle = [exterior_mapping[t] for t in triangle]

            rx = e1 * triangle[2][0] + e2 * triangle[0][0] + e3 * triangle[1][0]
            ry = e1 * triangle[2][1] + e2 * triangle[0][1] + e3 * triangle[1][1]

            tx = (dx + rx)/2
            ty = (dy + ry)/2
            
            new_vertices.append((tx,ty))

        return Mesh(new_vertices, mesh.faces, mesh.dim)

    @staticmethod
    def triangle_area(p1, p2, p3):
        mat = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
        return abs(0.5 * np.linalg.det(mat))
    
    @staticmethod
    def rbf(v):
        v = v / 20.0
        if (v > 1):
            return 0
        return (1 - v) * (1 - v)


