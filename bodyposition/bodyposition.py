from loguru import logger
import numpy as np
import imma.image
import bodyposition.CT_regression_tools
# from . import CT_regression_tools
# import CT_regression_tools
import os
from pathlib import Path

path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

models = {}

urllib = {
    "sagittal": "https://drive.google.com/uc?export=download&id=1wPq2tA-0k12cQutt7kZzv74rQ_WSLkCP",
    "coronal": "https://drive.google.com/uc?export=download&id=1tORaZ9LMpOp5MQS38LZ3WV5YrWc-ziuy",
    "surface": "https://drive.google.com/uc?export=download&id=1zi2iCeK0My2GB0ymOSJzxQhXnhtLwn3N",
    "bones": "https://drive.google.com/uc?export=download&id=10Xp13myCq7ZFkU3Px9JRvUqLw12CZZ_L",
    "fatless": "https://drive.google.com/uc?export=download&id=1Cj26jZETnhNJ0RlEtzL2C88PZkWU6Hv-",
    "liver": "https://drive.google.com/uc?export=download&id=17Paj-WHdBmLNk-_4QrRWKpUdInCnu4z-",
    "spleen": "https://drive.google.com/uc?export=download&id=1P1U8iD_rlaYzMvB6e5LgydpK5_3K5UNL",
    "lungs": "https://drive.google.com/uc?export=download&id=1U6qCs8DlmSpDykWSv0UrAv62xR-tntlq",
}

class BodyPosition:

    def __init__(
        self, data3d, voxelsize_mm, imshape=256, ):
        self.imshape = 256
        self.sdf_type = ''
        self.models = models
        self.working_vs = np.asarray([1.] * 3)
        zsize = int(data3d.shape[0] * voxelsize_mm[0] / self.working_vs[0])
        self.data3dr = imma.image.resize_to_shape(data3d, [zsize, imshape, imshape])
        self.orig_shape = data3d.shape

    def dist_to_sagittal(self):
        sdf_type = 'sagittal'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)

    def dist_to_coronal(self):
        sdf_type = 'coronal'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def dist_to_surface(self):
        sdf_type = 'surface'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def dist_to_bones(self):
        sdf_type = 'bones'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def dist_to_fatless(self):
        sdf_type = 'fatless'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def dist_to_liver(self):
        sdf_type = 'liver'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def dist_to_spleen(self):
        sdf_type = 'spleen'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def dist_to_lungs(self):
        sdf_type = 'lungs'
        if sdf_type not in models:
            model = self._get_model(sdf_type)
            self.model = model
        
        predictions = self._predict()
        
        return self._resize_to_orig_shape(predictions)
    
    def _resize_to_orig_shape(self, data):
        return imma.image.resize_to_shape(data, self.orig_shape)
    
    def _get_devel_model_path(self, sdf_type):
        model_path = path_to_script.parent.parent / f"bodyposition-models/final_sdf_unet_{sdf_type}.h5"
        return model_path
    
    def download_model(self, sdf_type):
        import requests

        model_path = self._get_devel_model_path(sdf_type)
        url = urllib[sdf_type]
        
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            # logger.debug(f"Downloading from '{url}' to {str(model_path)}")
            r = requests.get(url, allow_redirects=True)
            open(model_path, "wb").write(r.content)
        return model_path
    
    def _get_model(self, sdf_type):
        from tensorflow.keras.models import load_model
        if sdf_type in models:
            model = models[sdf_type]
        else:
            model_path = self._get_devel_model_path(sdf_type)
            if not model_path.exists():
                model_path = self.download_model(sdf_type)
            model = load_model(model_path)
            models[sdf_type] = model
        return model
    
    def _predict(self):
        data = CT_regression_tools.normalize(self.data3dr)
        data = self._resize(data)
        data = np.asarray(data).reshape(np.asarray(data).shape[0], self.imshape, self.imshape, 1)
        
        predictions = self.model.predict(data)
        
        predictions = np.asarray(predictions).reshape(np.asarray(predictions).shape[0], self.imshape, self.imshape)
        self.predictions = predictions
        return predictions
    
    def _resize(self, data):
        data2 = []
        for i in range(len(data)):
            data2.append(CT_regression_tools.resize(data[i], self.imshape))
        return data2
    
    def get_data(self, dataset, scannum):
        from bodyposition import seg
        ss, data3d, voxelsize = seg.read_scan(dataset, scannum)
        return data3d
        
if __name__ == "__main__":
    pass