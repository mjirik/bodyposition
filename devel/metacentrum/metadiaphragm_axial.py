import sys
print(sys.path)

from bodyposition import sdf_unet256, sdf_unet256_tensorboard
import time

sdf_type='diaphragm_axial'

start_time = time.time()

sdf_unet256_tensorboard.train(
    imshape=256,
    sdf_type=sdf_type,
    skip_h5=False,
    batch_size=16,
    epochs=200,
    filename_prefix='',
    validation_ids = [19, 39],
    test_ids = [20, 40],
    n_data = 40,
)

print(f"{sdf_type}: Training time: {time.time() - start_time} seconds")