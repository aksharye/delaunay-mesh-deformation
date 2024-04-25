from mesh import Mesh
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# Currently only 2D support
# IDW Method

class DTIDW_Mesh_Deform:
    @staticmethod
    def visualize_delaunay(vertices):
        vertices = np.array(vertices)
        tri = Delaunay(vertices)
        plt.triplot(vertices[:,0], vertices[:,1], tri.simplices)
        plt.plot(vertices[:,0], vertices[:,1], 'o')
        plt.show()

    @staticmethod
    def deform(mesh, exterior, exterior_mapping, is_rotation=False):
        tri = Delaunay(exterior)
        simplices = tri.simplices

        v_tris = tri.find_simplex(mesh.vertices)

        new_vertices = []
        for i in range(len(mesh.vertices)):
            
            x = mesh.vertices[i][0]
            y = mesh.vertices[i][1]
            
            if ((x,y) in exterior_mapping):
                new_vertices.append(exterior_mapping[(x,y)])
                continue;
            
            triangle = simplices[v_tris[i]]
            triangle = [(exterior[triangle[0]][0], exterior[triangle[0]][1]), (exterior[triangle[1]][0], exterior[triangle[1]][1]), (exterior[triangle[2]][0], exterior[triangle[2]][1])]
            
            numerator = (0,0)
            denominator = 0.0
            for i in range(3):
                if (triangle[i][0] == exterior_mapping[triangle[i]][0] and triangle[i][1] == exterior_mapping[triangle[i]][1]):
                    denominator += (1.0 / np.linalg.norm(np.subtract((x,y),triangle[i])))
                else:
                    numerator += (tuple(np.subtract(exterior_mapping[triangle[i]], triangle[i]))/ (np.linalg.norm(np.subtract((x,y),triangle[i])))) 
                    denominator += (1.0 / (np.linalg.norm(np.subtract((x,y),triangle[i])) ))
            
            print(denominator)
            
            IDW = numerator / denominator
            
            tx = x + IDW[0]
            ty = y + IDW[1]
            
            new_vertices.append((tx,ty))

        return Mesh(new_vertices, mesh.faces, mesh.dim)

    @staticmethod
    def triangle_area(p1, p2, p3):
        mat = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
        return abs(0.5 * np.linalg.det(mat))



