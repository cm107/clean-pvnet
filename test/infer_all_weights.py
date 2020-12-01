from clean_pvnet.infer.pvnet_inferer import PVNetInferer
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.path_utils import get_all_files_of_extension, get_rootname_from_path
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir, file_exists
from tqdm import tqdm

img_dir = '/home/clayton/workspace/prj/data/misc_dataset/darwin_datasets/nihonbashi2/organized'
linemod_dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')

coco_dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201017_robot_camera/combined/output.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201017_robot_camera/combined'
)
linemod_ann_sample = linemod_dataset.annotations[0]
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
corner_3d = linemod_ann_sample.corner_3d
K = linemod_ann_sample.K
linemod_image_sample = linemod_dataset.images[0]
dsize = (linemod_image_sample.width, linemod_image_sample.height)

weights_dir = '/home/clayton/workspace/git/clean-pvnet/data/model/pvnet/custom'
weight_path_list = get_all_files_of_extension(weights_dir, 'pth')
weight_path_list.sort()
infer_data_dump_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201017_robot_camera/infer_dump'
make_dir_if_not_exists(infer_data_dump_dir)
# delete_all_files_in_dir(infer_data_dump_dir, ask_permission=True)
weights_pbar = tqdm(total=len(weight_path_list), unit='weight(s)')
for weight_path in weight_path_list:
    rootname = get_rootname_from_path(weight_path)
    weights_pbar.set_description(rootname)
    pred_dump_path = f'{infer_data_dump_dir}/{rootname}.json'
    if file_exists(pred_dump_path):
        weights_pbar.update()
        continue
    inferer = PVNetInferer(weight_path=weight_path)
    inferer.infer_coco_dataset(
        dataset=coco_dataset,
        kpt_3d=kpt_3d,
        corner_3d=corner_3d,
        K=K,
        blackout=True,
        dsize=dsize,
        pred_dump_path=pred_dump_path
    )
    weights_pbar.update()