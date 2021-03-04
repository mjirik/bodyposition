from loguru import logger
import numpy as np
import imma.image

models = {}

class BodyPosition:

    def __init__(
        self, data3d, voxelsize_mm, imshape=256, ):
        self.imshape = 256
        self.sdf_type = ''
        self.models = models
        self.working_vs = np.asarray([1.] * 3)
        self.data3dr = imma.image.resize_to_mm(data3d, voxelsize_mm, self.working_vs)
        self.orig_shape = data3d.shape

    def get_dist_to_sagittal(self):
        data = self.data3d
        sdf_type = 'sagittal'
        if self.model == None:
            model = self._get_model(sdf_type)
            self.model = model
        predictions = model.predict(data)
        predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
        self.predictions = predictions
        return self._resize_to_orig_shape(predictions)

    def _resize_to_orig_shape(self, data):
        return imma.image.resize_to_shape(data, self.orig_shape)

    def get_dist_to_coronal(self, data):
        sdf_type = 'coronal'
        model = self._get_model(sdf_type)
    
    def get_dist_to_surface(self, data):
        sdf_type = 'surface'
        model = self._get_model(sdf_type)
    
    def _get_model(self, sdf_type):
        from tensorflow.keras.models import load_model
        if sdf_type in models:
            model = models[sdf_type]
        else:
            model = load_model(f"sdf_unet_{sdf_type}.h5")
            models[sdf_type] = model
        return model
    
    def get_data(self, dataset, scannum):
        from bodyposition import seg
        ss, data3d, voxelsize = seg.read_scan(dataset, scannum)
        return data3d
        
if __name__ == "__main__":
    pass