from mesh import Mesh
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# Currently only 2D support

class DTRBFAD_Mesh_Deform:
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


            A = DTRBFAD_Mesh_Deform.triangle_area(triangle[0], triangle[1], triangle[2])

            e1 = DTRBFAD_Mesh_Deform.triangle_area(triangle[0], triangle[1], (x,y)) / A
            e2 = DTRBFAD_Mesh_Deform.triangle_area(triangle[1], triangle[2], (x,y)) / A
            e3 = DTRBFAD_Mesh_Deform.triangle_area(triangle[0], triangle[2], (x,y)) / A
            
            triangle_weights = []
            for i in range(3):
                if (triangle[i][0] == exterior_mapping[triangle[i]][0] and triangle[i][1] == exterior_mapping[triangle[i]][1]):
                    triangle_weights.append(0.5);
                else:
                    triangle_weights.append(0);
                    
            triangle = [exterior_mapping[t] for t in triangle]

            dx = []
            dy = []
            for j in range(3):
                dx.append(triangle[j][0] - exterior_mapping[triangle[j]][0])
                dy.append(triangle[j][1] - exterior_mapping[triangle[j]][1])

            dx = np.array(dx)
            dy = np.array(dy)

            M = []
            for j in range(3):
                row = []
                for k in range(3):
                    row.append(DTRBFAD_Mesh_Deform.rbf(np.linalg.norm(np.array(triangle[j]) - np.array(triangle[k]))))
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
                A.append(DTRBFAD_Mesh_Deform.rbf(np.linalg.norm(np.array((x,y)) - np.array(triangle[j]))))
            A = np.array(A)
            
            dx = np.dot(A, alpha_x)
            dy = np.dot(A, alpha_y)

            r = e1 * triangle_weights[2] + e2 * triangle_weights[0] + e3 * triangle_weights[1]
            damp = DTRBFAD_Mesh_Deform.damp(r)
            
            tx = x + damp * dx
            ty = y + damp * dy

            new_vertices.append((tx,ty))

        return Mesh(new_vertices, mesh.faces, mesh.dim)
    
    @staticmethod
    def deform_rotation(mesh, exterior, rotation_mapping):
        tri = Delaunay(exterior)
        simplices = tri.simplices

        v_tris = tri.find_simplex(mesh.vertices)

        exterior_mapping = {}

        for p in rotation_mapping:
            theta = rotation_mapping[p]
            rotate = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            exterior_mapping[p] = tuple(np.dot(rotate, p))
            
        new_vertices = []
        for i in range(len(mesh.vertices)):
            x = mesh.vertices[i][0]
            y = mesh.vertices[i][1]
            
            triangle = simplices[v_tris[i]]
            triangle = [(exterior[triangle[0]][0], exterior[triangle[0]][1]), (exterior[triangle[1]][0], exterior[triangle[1]][1]), (exterior[triangle[2]][0], exterior[triangle[2]][1])]


            A = DTRBFAD_Mesh_Deform.triangle_area(triangle[0], triangle[1], triangle[2])

            e1 = DTRBFAD_Mesh_Deform.triangle_area(triangle[0], triangle[1], (x,y)) / A
            e2 = DTRBFAD_Mesh_Deform.triangle_area(triangle[1], triangle[2], (x,y)) / A
            e3 = DTRBFAD_Mesh_Deform.triangle_area(triangle[0], triangle[2], (x,y)) / A
            
            triangle_weights = []
            for i in range(3):
                if (triangle[i][0] == exterior_mapping[triangle[i]][0] and triangle[i][1] == exterior_mapping[triangle[i]][1]):
                    triangle_weights.append(0.5);
                else:
                    triangle_weights.append(0);
        
            rz = []
            for j in range(3):
                rz.append(rotation_mapping[triangle[j]])

            rz = np.array(rz)

            M = []
            for j in range(3):
                row = []
                for k in range(3):
                    row.append(DTRBFAD_Mesh_Deform.rbf(np.linalg.norm(np.array(triangle[j]) - np.array(triangle[k]))))
                M.append(row)

            M = np.array(M)
            if np.linalg.det(M) != 0:
                Minv = np.linalg.inv(M)
            else:
                new_vertices.append((x,y))
                continue

            alpha_z = np.dot(Minv, rz)
            A = []
            for j in range(3):
                A.append(DTRBFAD_Mesh_Deform.rbf(np.linalg.norm(np.array((x,y)) - np.array(triangle[j]))))
            A = np.array(A)

            
            theta = np.dot(A, alpha_z)


            r = e1 * triangle_weights[2] + e2 * triangle_weights[0] + e3 * triangle_weights[1]
            damp = DTRBFAD_Mesh_Deform.damp(r)
            theta = damp * theta


            tx = x*np.cos(theta) + y*np.sin(theta)
            ty = y*np.cos(theta) - x*np.sin(theta)
            
            new_vertices.append((tx,ty))

        return Mesh(new_vertices, mesh.faces, mesh.dim)

    @staticmethod
    def triangle_area(p1, p2, p3):
        mat = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
        return abs(0.5 * np.linalg.det(mat))
    
    @staticmethod
    def damp(v):
        if (v > 1):
            return 0
        return (1 - v) * (1 - v) * (1 + v)
    
    @staticmethod
    def rbf(v):
        v = v / 10.0
        if (v > 1):
            return 0
        return (1 - v) * (1 - v)



