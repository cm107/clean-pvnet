from clean_pvnet.infer.pvnet_inferer import PVNetInferer
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset

inferer = PVNetInferer(
    weight_path='/home/clayton/workspace/git/clean-pvnet/data/model/pvnet/custom/99.pth',
    # weight_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_weights/194.pth'
)
img_dir = '/home/clayton/workspace/git/pvnet-rendering/test/renders1'
linemod_dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')
# inferer.infer_linemod_dataset(dataset=linemod_dataset, img_dir=img_dir, blackout=True)

coco_dataset = COCO_Dataset.load_from_path(
    json_path='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200214/1/coco-data/fixed_HSR-coco.json',
    img_dir='/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200214/1/coco-data'
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
    dsize=dsize
)

# TODO: Create COCO -> Linemod dataset conversion method