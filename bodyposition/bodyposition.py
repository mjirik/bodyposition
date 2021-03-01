from loguru import logger
import numpy as np

class BodyPosition(
    
):
    def __init__(
        self, imshape, head_first=True
        ):
        
        def get_dist_to_sagittal(data):
            sdf_type = 'sagittal'
            model = get_model(sdf_type)
            predictions = model.predict(data)
            predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
            return predictions
        
        def get_dist_to_coronal(data):
            sdf_type = 'coronal'
            model = get_model(sdf_type)
        
        def get_dist_to_surface(data):
            sdf_type = 'surface'
            model = get_model(sdf_type)
        
        def get_model(sdf_type):
            from tensorflow.keras.models import load_model
            model = load_model(f"sdf_unet_{sdf_type}.h5")
            return model
        
        def get_data(dataset, scannum):
            from bodyposition import seg
            ss, data3d, voxelsize = seg.read_scan(dataset, scannum)
            return data3d
        
if __name__ == "__main__":
    pass