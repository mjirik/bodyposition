from loguru import logger
import numpy as np
import imma.image
# import bodyposition.CT_regression_tools
from . import CT_regression_tools

models = {}

class BodyPosition:

    def __init__(
        self, data3d, voxelsize_mm, imshape=256, ):
        self.imshape = 256
        self.sdf_type = ''
        self.models = models
        self.working_vs = np.asarray([1.] * 3)
        # self.data3dr = imma.image.resize_to_shape(data3d, [int(data3d.shape[0] * voxelsize_mm[0] / self.working_vs[0]), imshape, imshape])
        self.data3dr = data3d
        self.orig_shape = [data3d.shape[0], self.imshape, self.imshape]

    def get_dist_to_sagittal(self):
        sdf_type = 'sagittal'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        data = self._resize()
        data = np.asarray(data).reshape(np.asarray(data).shape[0], self.imshape, self.imshape, 1)
        
        predictions = self.model.predict(data)
        
        predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
        self.predictions = predictions
        
        return self._resize_to_orig_shape(predictions)

    def get_dist_to_coronal(self):
        sdf_type = 'coronal'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        data = self._resize()
        data = np.asarray(data).reshape(np.asarray(data).shape[0], self.imshape, self.imshape, 1)
        
        predictions = self.model.predict(data)
        
        predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
        self.predictions = predictions
        
        return self._resize_to_orig_shape(predictions)
    
    def get_dist_to_surface(self):
        sdf_type = 'surface'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        data = self._resize()
        data = np.asarray(data).reshape(np.asarray(data).shape[0], self.imshape, self.imshape, 1)
        
        predictions = self.model.predict(data)
        
        predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
        self.predictions = predictions
        
        return self._resize_to_orig_shape(predictions)
    
    def _resize_to_orig_shape(self, data):
        return imma.image.resize_to_shape(data, self.orig_shape)
    
    def _get_model(self, sdf_type):
        from tensorflow.keras.models import load_model
        if sdf_type in models:
            model = models[sdf_type]
        else:
            model = load_model(f"sdf_unet_{sdf_type}.h5")
            models[sdf_type] = model
        return model
    
    def _resize(self):
        data2 = []
        for i in range(len(self.data3dr)):
            data2.append(CT_regression_tools.resize(self.data3dr[i], self.imshape))
        return data2
    
    def get_data(self, dataset, scannum):
        from bodyposition import seg
        ss, data3d, voxelsize = seg.read_scan(dataset, scannum)
        return data3d
        
if __name__ == "__main__":
    pass