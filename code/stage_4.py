import open3d as o3d
import numpy as np
import scipy
import time

# Meshless Deformations Based on Shape Matching
# cluster based deformation implementation as described in Section 4.4

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

ALPHA = 0.1
BETA = 0.6
H = 0.3
DAMPING = 0.999
FLOOR_LEVEL = -10.

NUM_CLUSTERS = 4

# load mesh
# low res 144 points
# high res 1466 points
mesh = o3d.io.read_triangle_mesh("data/rectangle.obj", True)
mesh.compute_vertex_normals()

# set floor
whd = [20., 20., 2.]
whd[Z_AXIS] = 1.
mesh_box = o3d.geometry.TriangleMesh.create_box(width=whd[0],
                                                height=whd[1],
                                                depth=whd[2])
mesh_box.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_box.vertices) + np.asarray([-10, -10, FLOOR_LEVEL]))
mesh_box.compute_vertex_normals()
mesh_box.paint_uniform_color([0.5, 0.5, 0.1])

# ensure mesh is above floor
x_inits = np.asarray(mesh.vertices).copy()
min_z = np.min(x_inits[:, Z_AXIS])
if min_z < 0:
    x_inits[:, Z_AXIS] += -min_z + 20.
    mesh.vertices = o3d.utility.Vector3dVector(x_inits)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.add_geometry(mesh_box)

# initialize velocity vectors
velocity = np.zeros_like(x_inits)

start_point = min(x_inits[:, 0])
end_point = max(x_inits[:, 0])
# fixed points
fixed_point_arr = np.array([0 if ((row[0] <= start_point + 1)  or (row[0] >= end_point - 1)) else 1 for row in x_inits])
fixed_filter = np.zeros((x_inits.shape[0], 3))
fixed_filter[:, 0] = fixed_point_arr
fixed_filter[:, 1] = fixed_point_arr
fixed_filter[:, 2] = fixed_point_arr

# create clusters
clusters = []

cluster_length = (2 / (NUM_CLUSTERS + 1)) * (end_point - start_point)
offset = cluster_length / 2

for i in range(NUM_CLUSTERS):
    begin = start_point + i*offset
    end = begin + cluster_length
    
    bool_arr = np.array([((begin<=row[0]) and (row[0]<=end)) for row in x_inits])
    clusters.append(bool_arr)

start_time = time.time()
frames = 0

while(True):

    x = np.asarray(mesh.vertices).copy()

    # floor external force
    z_vals = x[:, Z_AXIS]
    a = np.where(z_vals > FLOOR_LEVEL, 0., -1.)
    b = np.where(z_vals > FLOOR_LEVEL, z_vals, 0.)
    vel_vec = np.zeros_like(x, dtype=x.dtype)
    vel_vec[:, Z_AXIS] = a

    velocity += velocity * vel_vec

    x[:, Z_AXIS] = a + b

    for c in clusters:

        x_cluster = x[c]
        x_inits_cluster = x_inits[c]
        
        t_0 = np.mean(x_inits_cluster, axis=0)
        q = x_inits_cluster - t_0

        t = np.mean(x_cluster, axis=0)
        p = x_cluster - t

        A_pq = np.dot(p.T, q)
        S_inv = np.linalg.pinv(scipy.linalg.sqrtm(np.dot(A_pq.T, A_pq)))

        # rotation matrix
        R = np.dot(A_pq, S_inv)

        R_tilda = np.zeros((3, 9))
        R_tilda[:3, :3] = R 

        qx = q[:, 0]
        qy = q[:, 1]
        qz = q[:, 2]
        q_tilda = np.array([qx, qy, qz, qx*qx, qy*qy, qz*qz, qx*qy, qy*qz, qz*qx]).T

        Aqq_tilda = np.dot(q_tilda.T, q_tilda)
        Aqq_tilda = np.linalg.pinv(Aqq_tilda)

        Apq_tilda = np.dot(p.T, q_tilda)

        A_tilda = np.dot(Apq_tilda, Aqq_tilda)

        # goal points 
        g = np.dot((BETA*A_tilda + (1-BETA)*R_tilda), q_tilda.T).T + t

        velocity[c] += ALPHA * (g - x_cluster) / H

    # include gravity 
    f_ext = np.zeros((1, 3), dtype=velocity.dtype)
    f_ext[:, Z_AXIS] = -0.1
    velocity += H * f_ext
    velocity *= DAMPING
    
    # fix fixed points
    velocity *= fixed_filter

    # update position
    mesh.vertices = o3d.utility.Vector3dVector(x + H * velocity)

    frames += 1
    print( str(frames) + " frames")
    print("--- %s seconds ---" % (time.time() - start_time))

    vis.update_geometry(mesh)
    if not vis.poll_events():
        break
    vis.update_renderer()