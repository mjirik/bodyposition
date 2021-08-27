import numpy as np
import io3d
import sed3

import bodyposition as bpo
print(bpo.__file__)

data3d_orig = io3d.read_dataset("3Dircadb1", "data3d", 1)
bp = bpo.BodyPosition(data3d_orig['data3d'], data3d_orig['voxelsize_mm'])
sdf_surface = bp.dist_to_surface()

sed3.show_slices(np.asarray(data3d_orig['data3d'][0:-1]), np.asarray(sdf_surface[0:-1]), slice_step=10, axis=0)