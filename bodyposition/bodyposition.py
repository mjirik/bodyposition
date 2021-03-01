from loguru import logger
import numpy as np

class BodyPosition(
    
):
    def __init__(
        self, imshape, head_first=True
        ):
        
        def get_dist_to_sagittal():
            sdf_type = 'sagittal'
        
        def get_dist_to_coronal():
            sdf_type = 'coronal'
        
        def get_dist_to_surface():
            sdf_type = 'surface'
        
        def get_model(sdf_type):
            from tensorflow.keras.models import load_model
            model = load_model(f"sdf_unet_{sdf_type}.h5")
            return model
        
        def get_data(dataset, scannum):
            from bodyposition import seg
            ss, data3d, voxelsize = seg.read_scan(dataset, scannum)
            return data3d