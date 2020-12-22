import numpy as np
from clean_pvnet.infer.pvnet_inferer import PVNetFrameResultList
from common_utils.path_utils import get_all_files_of_extension, get_filename
from common_utils.file_utils import make_dir_if_not_exists
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from tqdm import tqdm

linemod_dataset = Linemod_Dataset.load_from_path('/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/combined/train.json')
linemod_ann_sample = linemod_dataset.annotations[0]
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
corner_3d = linemod_ann_sample.corner_3d
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
# old_K = linemod_ann_sample.K
K = np.array([517.799858, 0.000000, 303.876287, 0.000000, 514.807834, 238.157119, 0.000000, 0.000000, 1.000000]).reshape(3,3)
distortion = np.array([0.001647, -0.105636, -0.002094, -0.006446, 0.000000])

# frame_result_list = PVNetFrameResultList.load_from_path(f'{infer_data_dump}/{model_name}_infer.json')
# pbar = tqdm(total=len(frame_result_list), unit='frame(s)')
# for result in frame_result_list:
#     for pred in result.pred_list:
#         pred.recalculate(
#             gt_kpt_3d=kpt_3d,
#             corner_3d=corner_3d,
#             K=K,
#             distortion=np.array([0.001647, -0.105636, -0.002094, -0.006446, 0.000000]),
#             units_per_meter=1.0
#         )
#     pbar.update()
# pbar.close()
# frame_result_list.save_to_path(f'{infer_data_dump}/{model_name}_infer0.json', overwrite=True)


infer_data_dump_dir = 'infer_data_dump0'
infer_data_dump_paths = get_all_files_of_extension(infer_data_dump_dir, extension='json')
infer_data_dump_paths.sort()
fixed_infer_data_dump_dir = 'infer_data_dump'
make_dir_if_not_exists(fixed_infer_data_dump_dir)
pbar = tqdm(total=len(infer_data_dump_paths), unit='dump(s)', leave=True)
pbar.set_description('Fixing Inference Dumps')
for infer_data_dump_path in infer_data_dump_paths:
    results = PVNetFrameResultList.load_from_path(infer_data_dump_path)
    for result in results:
        for pred in result.pred_list:
            pred.recalculate(
                gt_kpt_3d=kpt_3d,
                corner_3d=corner_3d,
                K=K,
                distortion=distortion,
                units_per_meter=1.0
            )
    results.save_to_path(f'{fixed_infer_data_dump_dir}/{get_filename(infer_data_dump_path)}', overwrite=True)
    pbar.update()
pbar.close()