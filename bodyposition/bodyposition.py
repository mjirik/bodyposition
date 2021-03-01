from loguru import logger
import numpy as np

class BodyPosition(
    
):
    def __init__(
        self, imshape=256
        ):
        self.imshape = 256
        self.sdf_type = ''
        self.model = None
        
    def get_dist_to_sagittal(self, data):
        sdf_type = 'sagittal'
        if self.model = None:
            model = get_model(sdf_type)
            self.model = model
        predictions = model.predict(data)
        predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
        self.predictions = predictions
        return predictions
    
    def get_dist_to_coronal(self, data):
        sdf_type = 'coronal'
        model = get_model(sdf_type)
    
    def get_dist_to_surface(self, data):
        sdf_type = 'surface'
        model = get_model(sdf_type)
    
    def get_model(self, sdf_type):
        from tensorflow.keras.models import load_model
        model = load_model(f"sdf_unet_{sdf_type}.h5")
        return model
    
    def get_data(self, dataset, scannum):
        from bodyposition import seg
        ss, data3d, voxelsize = seg.read_scan(dataset, scannum)
        return data3d
        
if __name__ == "__main__":
    pass