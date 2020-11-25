import numpy as np
from clean_pvnet.infer.pvnet_inferer import PVNetFrameResultList
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from tqdm import tqdm

img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/nihonbashi2/organized'
linemod_dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')
linemod_ann_sample = linemod_dataset.annotations[0]
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
corner_3d = linemod_ann_sample.corner_3d
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
# old_K = linemod_ann_sample.K

model_name = '20201119_epoch499'
robot_camera_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201017_robot_camera'
infer_data_dump = f'{robot_camera_dir}/infer_dump'

K = np.array([517.799858, 0.000000, 303.876287, 0.000000, 514.807834, 238.157119, 0.000000, 0.000000, 1.000000]).reshape(3,3)

frame_result_list = PVNetFrameResultList.load_from_path(f'{infer_data_dump}/{model_name}_infer.json')
pbar = tqdm(total=len(frame_result_list), unit='frame(s)')
for result in frame_result_list:
    for pred in result.pred_list:
        pred.recalculate(
            gt_kpt_3d=kpt_3d,
            corner_3d=corner_3d,
            K=K,
            # camera_translation=np.array([0.110, 0.060, 0.970]),
            # camera_quaternion=np.array([-0.500, 0.500, -0.500, 0.500]),
            distortion=np.array([0.001647, -0.105636, -0.002094, -0.006446, 0.000000]),
            units_per_meter=1.0
        )
    pbar.update()
pbar.close()
frame_result_list.save_to_path(f'{infer_data_dump}/{model_name}_infer0.json', overwrite=True)