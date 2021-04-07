import pytest
import io3d
import sed3
import bodyposition.bodyposition as bpo
import numpy as np
import h5py
import bodyposition.CT_regression_tools

def test_import():

    data3d_orig = io3d.read_dataset("3Dircadb1", "data3d", 1)
    bp = bpo.BodyPosition(bodyposition.CT_regression_tools.normalizescan2(data3d_orig['data3d']), data3d_orig['voxelsize_mm'])
    sdf_sagittal = bp.get_dist_to_sagittal()

    assert np.max(sdf_sagittal) > 0
    assert np.min(sdf_sagittal) < 0

    imsize_px = np.prod(data3d_orig['data3d'].shape)
    imsize_px_010 = imsize_px * 0.1
    assert np.sum(sdf_sagittal > 0) > imsize_px_010
    assert np.sum(sdf_sagittal < 0) > imsize_px_010
    
    # data2 = []
    # for i in range(len(data3d_orig['data3d'])):
    #     data2.append(CT_regression_tools.resize(data3d_orig['data3d'][i], 256))
    # sed3.show_slices(np.asarray(data2[0:-1]), np.asarray(sdf_sagittal[0:-1]), slice_step=10, axis=0)

if __name__ == "__main__":
    test_import()