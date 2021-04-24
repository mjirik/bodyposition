import seg
import numpy as np
from loguru import logger
import h5py
import skimage.io
import skimage
import skimage.transform
import CT_regression_tools
import matplotlib.pyplot as plt
# from bodynavigation.organ_detection import OrganDetection

def prepare_data(
    imshape=256,
    sdf_type='diaphragm_axial',
    # sdf_type='coronal',
    # sdf_type='sagittal',
    # sdf_type='surface',
    skip_h5=False,
    n_data=40,
    filename_prefix='',
    organ_detection_on=True,
):
    """

    :param imshape:
    :param sdf_type:
    :param skip_h5:
    :param n_data:
    :param filename_prefix: used to prevent rewriting the files during testing
    :return:
    """


    c=0
    for i in range(20,n_data):
        if i <= 19:
            ss, data, voxelsize = seg.read_scan("3Dircadb1", i+1)
        else:
            ss, data, voxelsize = seg.read_scan("sliver07", i-19)
        
        X_train = np.empty([len(data), imshape, imshape], dtype=np.float) # more efficient

        # for j in range(n_data):
        for j in range(data.shape[0]):

                img = CT_regression_tools.resize(data[j], imshape)
                img = CT_regression_tools.normalize(img)
                X_train[j] = img

        if organ_detection_on:
            from bodynavigation.organ_detection import OrganDetection
            obj = OrganDetection(data, voxelsize)
            Y_train1 = obj.getBones()
            Y_train2 = obj.getFatlessBody()
            Y_train1 = skimage.transform.resize(np.asarray(Y_train1), [Y_train1.shape[0], imshape, imshape], preserve_range = True)
            Y_train2 = skimage.transform.resize(np.asarray(Y_train2), [Y_train2.shape[0], imshape, imshape], preserve_range = True)
        else:
            Y_train = eval(f"ss.dist_to_{sdf_type}()")
            Y_train = skimage.transform.resize(np.asarray(Y_train), [Y_train.shape[0], imshape, imshape], preserve_range = True)
        
        # sed3.show_slices(np.asarray(X_train[0:50]), np.asarray(Y_train[0:50]), slice_step=10, axis=2)
        # plt.show()

        if not skip_h5:
            if organ_detection_on:
                with h5py.File(f'{filename_prefix}sdf_bones{imshape}.h5', 'a') as h5f:
                    logger.debug(f"X_train={X_train.dtype}")
                    h5f.create_dataset(f'scan_{i}', data=X_train)
                    h5f.create_dataset(f'label_{i}', data=Y_train1)
                with h5py.File(f'{filename_prefix}sdf_fatless{imshape}.h5', 'a') as h5f:
                    logger.debug(f"X_train={X_train.dtype}")
                    h5f.create_dataset(f'scan_{i}', data=X_train)
                    h5f.create_dataset(f'label_{i}', data=Y_train2)
            else:
                with h5py.File(f'{filename_prefix}sdf_{sdf_type}{imshape}.h5', 'a') as h5f:
                    logger.debug(f"X_train={X_train.dtype}")
                    h5f.create_dataset(f'scan_{i}', data=X_train)
                    h5f.create_dataset(f'label_{i}', data=Y_train)
            c += 1
            logger.info(f'Scan n.{c} saved. i={i}')


if __name__ == "__main__":
    # this will be skipped if file is imported but it will work if file is called from commandline
    prepare_data()