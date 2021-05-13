import h5py
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import sed3
import time
import io3d
from imma.volumetry_evaluation import compare_volumes
import exsu
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

def compare(
    imshape=256,
    # sdf_type='coronal',
    # sdf_type='sagittal',
    # sdf_type='surface',
    # sdf_type='bones',
    # sdf_type='fatless',
    sdf_type='liver',
    # sdf_type='spleen',
    # sdf_type='lungs',
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
    
    import bodynavigation
    from bodynavigation.organ_detection import OrganDetection
    import bodyposition

    pth = Path(__file__).parent.parent
    outputdir = pth / "experiments/"
    commonsheet = outputdir / "experiments.xlsx"
    report = exsu.report.Report(outputdir=outputdir, additional_spreadsheet_fn=commonsheet, check_version_of=["numpy", "scipy"])

    organ_detection_on = False
    if sdf_type == "liver" or sdf_type == "spleen" or sdf_type == "lungs" or sdf_type == "fatless" or sdf_type == "bones":
        organ_detection_on = True    
    
    data3d_orig = io3d.read_dataset(dataset, "data3d", scannum, orientation_axcodes='SPL')
    voxelsize = data3d_orig["voxelsize_mm"]
    
    # BODYNAVIGATION
    
    start_time = time.time()
    
    if not organ_detection_on:
        ss = bodynavigation.body_navigation.BodyNavigation(data3d_orig["data3d"], data3d_orig["voxelsize_mm"])
        sdf_bodynavigation = eval(f"ss.dist_to_{sdf_type}()")
    else:
        od = OrganDetection(data3d_orig["data3d"], voxelsize)
        if sdf_type == "liver":
            st = "Liver"
        if sdf_type == "spleen":
            st = "Spleen"
        if sdf_type == "lungs":
            st = "Lungs"
        if sdf_type == "fatless":
            st = "FatlessBody"
        if sdf_type == "bones":
            st = "Bones"
        sdf_bodynavigation = eval(f"od.get{st}()")
    time_bodynavigation = time.time() - start_time
    sed3.show_slices(np.asarray(data3d_orig["data3d"][0:-1]), np.asarray(sdf_bodynavigation[0:-1]), axis=0)

    
    # BODYPOSITION
    
    start_time = time.time()
    
    bpo = bodyposition.BodyPosition(data3d_orig["data3d"], data3d_orig['voxelsize_mm'])
    sdf_bodyposition = eval(f"bpo.dist_to_{sdf_type}()")
    time_bodyposition = time.time() - start_time
    sed3.show_slices(np.asarray(data3d_orig["data3d"][0:-1]), np.asarray(sdf_bodyposition[0:-1]), axis=0)
    
    # EVALUATION AND REPORT SAVING
    
    evaluation = compare_volumes(sdf_bodynavigation, sdf_bodyposition, voxelsize_mm=voxelsize)
    # logger.info(evaluation)
    
    report.add_cols_to_actual_row({
        "sdf type": sdf_type,
        "dataset": dataset,
        "scannum": scannum,
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
        "time_bodynavigation": time_bodynavigation,
        "time_bodyposition": time_bodyposition,
    })
    report.add_cols_to_actual_row(evaluation)
    report.finish_actual_row()
    report.dump()
    logger.info(f"Report finished - scan {scannum} (from {dataset}), {sdf_type} segmentation - BN {round(time_bodynavigation)} s, BP {round(time_bodyposition)} s")


if __name__ == "__main__":
    # this will be skipped if file is imported but it will work if file is called from commandline
    compare()