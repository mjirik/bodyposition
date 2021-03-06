import pytest
import io3d
import sed3
# import bodyposition.bodyposition as bpo
import bodyposition as bpo
import numpy as np
import h5py
# import bodyposition.CT_regression_tools
# import CT_regression_tools

def test_import():

    data3d_orig = io3d.read_dataset("3Dircadb1", "data3d", 1)
    bp = bpo.BodyPosition(data3d_orig['data3d'], data3d_orig['voxelsize_mm'])
    sdf_liver = bp.dist_to_lungs()

    assert np.max(sdf_liver) > 0
    assert np.min(sdf_liver) < 0

    imsize_px = np.prod(data3d_orig['data3d'].shape)
    imsize_px_010 = imsize_px * 0.1
    assert np.sum(sdf_liver >= 0) > imsize_px_010
    # assert np.sum(sdf_liver <= 0) > imsize_px_010
    
    sed3.show_slices(np.asarray(data3d_orig['data3d'][0:-1]), np.asarray(sdf_liver[0:-1]), slice_step=10, axis=0)

if __name__ == "__main__":
    test_import()
