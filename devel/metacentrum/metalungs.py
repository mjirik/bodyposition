import sys
print(sys.path)

from bodyposition import sdf_unet256, sdf_unet256_tensorboard
import time

sdf_type='lungs'

start_time = time.time()

sdf_unet256_tensorboard.train(
    imshape=256,
    sdf_type=sdf_type,
    skip_h5=False,
    batch_size=16,
    epochs=200,
    filename_prefix='',
    validation_ids = [19],
    test_ids = [20],
    n_data = 20,
)

print(f"{sdf_type}: Training time: {time.time() - start_time} seconds")