from clean_pvnet.infer.pvnet_inferer import PVNetInferer
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset

inferer = PVNetInferer(
    weight_path='/home/user/workspace/data/weights/499.pth'
)
img_dir = '/home/user/workspace/data/nihonbashi2/organized'
linemod_dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')

coco_dataset = COCO_Dataset.load_from_path(
    json_path='/home/user/workspace/data/20201017_robot_camera/combined/output.json',
    img_dir='/home/user/workspace/data/20201017_robot_camera/combined'
)
linemod_ann_sample = linemod_dataset.annotations[0]
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
corner_3d = linemod_ann_sample.corner_3d
K = linemod_ann_sample.K
linemod_image_sample = linemod_dataset.images[0]
dsize = (linemod_image_sample.width, linemod_image_sample.height)

inferer.infer_coco_dataset(
    dataset=coco_dataset,
    kpt_3d=kpt_3d,
    corner_3d=corner_3d,
    K=K,
    blackout=True,
    dsize=dsize,
    accumulate_pred_dump=False,
    video_save_path='test.avi', show_preview=True
)