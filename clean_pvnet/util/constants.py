import numpy as np

pvnet_mean = [0.485, 0.456, 0.406]
pvnet_std = [0.229, 0.224, 0.225]
pvnet_mean_reformatted = (
    np.array(pvnet_mean)
    .reshape(1, 1, 3)
    .astype(np.float32)
)
pvnet_std_reformatted = (
    np.array(pvnet_std)
    .reshape(1, 1, 3)
    .astype(np.float32)
)