import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

class Mesh:
    def __init__(self, vertices, faces, dim):
        self.vertices = vertices
        self.faces = faces
        self.dim = 2

    @staticmethod
    def faces_points_to_index(vertices, faces):
        vertex_to_index = {}
        for i in range(len(vertices)):
            vertex_to_index[tuple(vertices[i])] = i
        
        faces_by_index = []
        for f in faces:
            f_to_index = [vertex_to_index[tuple(v)] for v in f]
            faces_by_index.append(f_to_index)

        return faces_by_index

    def visualize(self, min_x = 0, min_y = 0, min_z = 0, max_x = 1, max_y = 1, max_z = 1, show_axis=True):
        vertices = self.vertices
        if (self.dim == 2):
            vertices = [[v[1], v[0], 0] for v in vertices]
        
        vertices = np.array(vertices)
        faces = np.array(self.faces)
        qualities = np.array(self.mesh_quality())

        C = np.array(range(0,100))

        fig = plt.figure()

        ax = fig.add_subplot(projection="3d")
        colors = plt.cm.jet(qualities)
        fig.colorbar(plt.cm.ScalarMappable(cmap = 'jet'), ax = ax)

        pc = art3d.Poly3DCollection(vertices[faces], facecolors=colors, edgecolor="black")
        ax.add_collection(pc)

        ax.set_xlim3d(min_x, max_x)
        ax.set_ylim3d(min_y, max_y)

        if (self.dim == 3):
            ax.set_zlim3d(0, max_z)
        
        if (self.dim == 2):
            ax.view_init(azim=0, elev=90)
            ax.set_zticks([])

        if not show_axis:
            ax.set_axis_off()
    
        fig.tight_layout()

        plt.show()

    def vertices_complement(self, vertices):
        return [v for v in self.vertices if v not in vertices]
    
    def face_quality(self, face):
        F = [self.vertices[f] for f in face]
        
        alphas = []
        sum = 0
        for k in range(4):
            Ak = []
            Ak.append([F[(k+1) % 4][0] - F[k % 4][0], F[(k+3) % 4][0] - F[k % 4][0]])
            Ak.append([F[(k+1) % 4][1] - F[k % 4][1], F[(k+3) % 4][1] - F[k % 4][1]])
            Ak = np.array(Ak)
            alpha_k = np.linalg.det(Ak)
            if (alpha_k < 0):
                return 0
            alphas.append(alpha_k)
            
            tensor_k = np.dot(Ak.T, Ak)
            sum += (np.sqrt(tensor_k[0][0] * tensor_k[1][1]))/alpha_k
            
            
            
        tau = (alphas[0] + alphas[2])/ 2.0
        size_metric = min(tau, 1.0 / tau)
        skew_metric = 4.0 / sum
        
        return np.sqrt(size_metric) * skew_metric

    def mesh_quality(self):
        self.qualities = []
        for f in self.faces:
            self.qualities.append(self.face_quality(f))
            
        return self.qualities
        
        

