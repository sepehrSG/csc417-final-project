import open3d as o3d
import numpy as np
import scipy

# Meshless Deformations Based on Shape Matching
# linear deformation implementation as described in Section 4.2

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

ALPHA = 0.01
BETA = 0.6
H = 0.3
DAMPING = 0.999

# load mesh
mesh = o3d.io.read_triangle_mesh("data/coarse_bunny.obj")
mesh.compute_vertex_normals()

# set floor
whd = [400., 400., 400.]
whd[Z_AXIS] = 1.
mesh_box = o3d.geometry.TriangleMesh.create_box(width=whd[0],
                                                height=whd[1],
                                                depth=whd[2])
mesh_box.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_box.vertices) + np.asarray([-200, -100, 0]))
mesh_box.compute_vertex_normals()
mesh_box.paint_uniform_color([0.5, 0.5, 0.1])

# ensure mesh is above floor
x_inits = np.asarray(mesh.vertices).copy()
min_z = np.min(x_inits[:, Z_AXIS])
if min_z < 0:
    x_inits[:, Z_AXIS] += -min_z + 20.
    mesh.vertices = o3d.utility.Vector3dVector(x_inits)

# calculate center of mass, assumin particle masses are all 1
t_0 = np.mean(x_inits, axis=0)
q = x_inits - t_0

# Aqq matrix calculation 
A_qq = np.linalg.pinv(np.dot(q.T, q))

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.add_geometry(mesh_box)

# initialize velocity vectors
velocity = np.zeros_like(x_inits)

while(True):

    x = np.asarray(mesh.vertices).copy()

    # floor external force
    z_vals = x[:, Z_AXIS]
    a = np.where(z_vals > -1, 0., -1.)
    b = np.where(z_vals > -1, z_vals, 0.)
    vel_vec = np.zeros_like(x, dtype=x.dtype)
    vel_vec[:, Z_AXIS] = a

    velocity += velocity * vel_vec

    x[:, Z_AXIS] = a + b

    t = np.mean(x, axis=0)
    p = x - t

    A_pq = np.dot(p.T, q)
    S_inv = np.linalg.pinv(scipy.linalg.sqrtm(np.dot(A_pq.T, A_pq)))

    # calcuate optimized linear transformation A
    A = np.dot(A_pq, A_qq)
    A = (1. / np.cbrt(np.linalg.det(A))) * A

    # rotation matrix
    R = np.dot(A_pq, S_inv)

    # goal points 
    g = np.dot((BETA*A  + (1-BETA)*R), q.T).T + t 

    # include gravity 
    f_ext = np.zeros((1, 3), dtype=velocity.dtype)
    f_ext[:, Z_AXIS] = -0.1
    velocity += ALPHA * (g - x) / H + H * f_ext
    velocity *= DAMPING

    # update position
    mesh.vertices = o3d.utility.Vector3dVector(x + H * velocity)

    vis.update_geometry(mesh)
    if not vis.poll_events():
        break
    vis.update_renderer()