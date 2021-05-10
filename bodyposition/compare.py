import h5py
import numpy as np
from loguru import logger
import sed3
import time
import seg
import io3d

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

def compare(
    imshape=256,
    # sdf_type='diaphragm_axial',
    # sdf_type='coronal',
    # sdf_type='sagittal',
    # sdf_type='surface',
    # sdf_type='bones',
    # sdf_type='fatless',
    # sdf_type='liver',
    # sdf_type='spleen',
    sdf_type='lungs',
    dataset = "3Dircadb1",
    scannum = 20,
):
    """Compare bodyposition and bodynavigation.

    Args:
        imshape (int, optional): Shape of dataset's images. Defaults to 256.
        sdf_type (str, optional): Type of sdf method. Can be: sagittal, coronal, surface, liver, spleen, lungs, bones, fatless or diaphragm_axial.
        dataset (string): "3Dircadb1" or "sliver07"
        scannum (int): Number of the scan this loads
    """
    
    import bodynavigation as bn
    import bodyposition as bp
    
    if sdf_type == "liver" or sdf_type == "spleen" or sdf_type == "lungs" or sdf_type == "fatless" or sdf_type == "bones":
        organ_detection_on = True
    
    start_time = time.time()
    data3d_orig = io3d.read_dataset(dataset, "data3d", scannum, orientation_axcodes='SPL')
    ss = bodynavigation.body_navigation.BodyNavigation(data3d_orig["data3d"], data3d_orig["voxelsize_mm"])
    sdf1 = eval(f"ss.dist_to_{sdf_type}()")
    time1 = time.time() - start_time
    # sed3.show_slices(np.asarray(data[0:-1]), np.asarray(sdf1[0:-1]), axis=0)
    
    start_time = time.time()
    data3d_orig = io3d.read_dataset(dataset, "data3d", scannum, orientation_axcodes='SPL')
    bpo = bp.Bodyposition(data3d_orig["data3d"])
    sdf2 = eval(f"bpo.get_dist_to_{sdf_type}()")
    time2 = time.time() - start_time
    # sed3.show_slices(np.asarray(data[0:-1]), np.asarray(sdf2[0:-1]), axis=0)
    
if __name__ == "__main__":
    # this will be skipped if file is imported but it will work if file is called from commandline
    compare()