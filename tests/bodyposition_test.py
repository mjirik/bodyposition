import bodyposition.bodyposition as bpo
import pytest
import io3d
# import bodyposition.bodyposition as bpo
import numpy as np


def test_import():

    data3d_orig = io3d.read_dataset("3Dircadb1", "data3d", 1)
    bp = bpo.BodyPosition(data3d_orig['data3d'], data3d_orig['voxelsize_mm'])
    sdf_sagittal = bp.get_dist_to_sagittal()

    assert np.max(sdf_sagittal) > 0
    assert np.min(sdf_sagittal) < 0

    imsize_px = np.prod(data3d_orig.data3d.shape)
    imsize_px_010 = imsize_px * 0.1
    assert np.sum(sdf_sagittal > 0) > imsize_px_010
    assert np.sum(sdf_sagittal < 0) > imsize_px_010
    # bodyposition.nejkyvnitrnimodul.fcn1()

if __name__ == "__main__":
    test_import()