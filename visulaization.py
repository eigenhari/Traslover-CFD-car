import numpy as np
import vtk
import os

root = './home/077bme053.hari/PINN/mlcfd_data/training_data'

def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output

name = '1b94aad142e6c2b8af9f38a1ee687286'

file_name_press = f'param0/{name}/quadpress_smpl.vtk'
file_name_velo = f'param0/{name}/hexvelo_smpl.vtk'
file_name_press = os.path.join(root, file_name_press)
file_name_velo = os.path.join(root, file_name_velo)

import numpy as np
import pyvista as pv

pv.start_xvfb()

reader = pv.get_reader(file_name_press)
mesh_gt_press = reader.read().rotate_y(90).rotate_x(90).rotate_z(270)

mesh_pred_press = mesh_gt_press.copy()
pred_press = np.load('sample/pred_press.npy')
mesh_pred_press.point_data['point_scalars'] = pred_press

points = np.load('sample/points_velo.npy')
gt_vectors = np.load('sample/gt_velo.npy')
pred_vectors = np.load('sample/pred_velo.npy')

max_x, min_x = max(points[:, 0]), min(points[:, 0])
max_y, min_y = max(points[:, 1]), min(points[:, 1])
max_z, min_z = max(points[:, 2]), min(points[:, 2])

grid_x, grid_y, grid_z = np.mgrid[min_x:max_x:128j, min_y:max_y:128j, min_z:max_z:128j]

from scipy.interpolate import griddata
gt_vectors = griddata(points, gt_vectors, (grid_x, grid_y, grid_z), method='linear')
gt_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
gt_grid["vectors"] = gt_vectors.reshape(-1, 3)

gt_streamlines = gt_grid.streamlines("vectors", progress_bar=True, integration_direction="both", n_points=500, source_radius=2)


pred_vectors = griddata(points, pred_vectors, (grid_x, grid_y, grid_z), method='linear')
pred_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
pred_grid["vectors"] = pred_vectors.reshape(-1, 3)

pred_streamlines = pred_grid.streamlines("vectors", progress_bar=True, integration_direction="both", n_points=500, source_radius=2)


p = pv.Plotter(shape=(1,2), window_size=[1024,640])
p.subplot(0,0)
p.add_mesh(mesh_pred_press, cmap="coolwarm", show_scalar_bar=False)
#p.add_scalar_bar('pressure', vertical=False, fmt='%.2f')
p.add_mesh(pred_streamlines.tube(radius=0.01).rotate_y(90).rotate_x(90).rotate_z(270), scalars="vectors", show_scalar_bar=False, line_width=2)
#p.add_scalar_bar('velocity', vertical=False, fmt='%.2f')
#p.add_bounding_box(line_width=2, color='black', opacity=0.2)
p.add_title('prediction')

p.subplot(0,1)
p.add_mesh(mesh_gt_press, cmap="coolwarm", show_scalar_bar=False)
p.add_scalar_bar('pressure', vertical=False, fmt='%.2f')
p.add_mesh(gt_streamlines.tube(radius=0.01).rotate_y(90).rotate_x(90).rotate_z(270), scalars="vectors", show_scalar_bar=False, line_width=2)
p.add_scalar_bar('velocity', vertical=False, fmt='%.2f')
#p.add_bounding_box(line_width=2, color='black', opacity=0.2)
p.add_title('ground truth')

#p.show_bounds(all_edges=True)

p.save_graphic(f'{name}.pdf')
