import pytest
import io3d
import sed3
# import bodyposition.bodyposition as bpo
import bodyposition as bpo
import numpy as np
from loguru import logger
import h5py
# import bodyposition.CT_regression_tools
# import CT_regression_tools

def test_import():

    # io3d.read()
    data3d_orig = io3d.read_dataset("3Dircadb1", "data3d", 1, orientation_axcodes="SPL")
    bp = bpo.BodyPosition(data3d_orig['data3d'], data3d_orig['voxelsize_mm'])
    sed3.show_slices(np.asarray(data3d_orig['data3d'][0:-1]), slice_step=10, axis=0)
    sdf = bp.dist_to_lungs()

    assert np.max(sdf) > 0
    assert np.min(sdf) < 0

    im_volume_px = np.prod(data3d_orig['data3d'].shape)

    sed3.show_slices(np.asarray(data3d_orig['data3d'][0:-1]), np.asarray(sdf[0:-1] > 0), slice_step=10, axis=0)
    organ_volume_px = np.sum(sdf >= 0)
    logger.debug(im_volume_px)
    logger.debug(organ_volume_px)
    assert organ_volume_px > (im_volume_px * 0.03)

if __name__ == "__main__":
    test_import()
