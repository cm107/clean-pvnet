import numpy as np
from common_utils.file_utils import file_exists, dir_exists
from common_utils.path_utils import recursively_get_all_filepaths_of_extension, \
    get_rootname_from_path, get_dirpath_from_filepath
from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.linemod.objects import Linemod_Dataset
from clean_pvnet.infer.pvnet_inferer import infer_tests_pvnet

weight_paths = [
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/99.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/199.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/299.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/399.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/499.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/599.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/699.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/799.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/899.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/999.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/99.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/149.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/199.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/249.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/299.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/399.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/499.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/599.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/99.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/199.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/299.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/399.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/499.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/599.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/699.pth',
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/darwin20210105/799.pth',
]
model_names = [
    # 'pvnet20201119-epoch99',
    # 'pvnet20201119-epoch199',
    # 'pvnet20201119-epoch299',
    # 'pvnet20201119-epoch399',
    # 'pvnet20201119-epoch499',
    # 'pvnet20201119-epoch599',
    # 'pvnet20201119-epoch699',
    # 'pvnet20201119-epoch799',
    # 'pvnet20201119-epoch899',
    # 'pvnet20201119-epoch999',
    # 'pvnet20201209-epoch99',
    # 'pvnet20201209-epoch149',
    # 'pvnet20201209-epoch199',
    # 'pvnet20201209-epoch249',
    # 'pvnet20201209-epoch299',
    # 'pvnet20201209-epoch399',
    # 'pvnet20201209-epoch499',
    # 'pvnet20201209-epoch599',
    'pvnet-darwin20210105-epoch99',
    'pvnet-darwin20210105-epoch199',
    'pvnet-darwin20210105-epoch299',
    'pvnet-darwin20210105-epoch399',
    'pvnet-darwin20210105-epoch499',
    'pvnet-darwin20210105-epoch599',
    'pvnet-darwin20210105-epoch699',
    'pvnet-darwin20210105-epoch799',
]
test_root_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201017_robot_camera'
csv_paths = recursively_get_all_filepaths_of_extension(test_root_dir, extension='csv')
test_names, datasets = [], []
for csv_path in csv_paths:
    test_name = get_rootname_from_path(csv_path)
    img_dir = f'{get_dirpath_from_filepath(csv_path)}/images'
    assert dir_exists(img_dir), f"Couldn't find image directory: {img_dir}"
    ann_path = f'{img_dir}/output.json'
    if not file_exists(ann_path):
        continue
    dataset = COCO_Dataset.load_from_path(ann_path, img_dir=img_dir)
    test_names.append(test_name)
    datasets.append(dataset)

linemod_dataset = Linemod_Dataset.load_from_path(f'/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/combined/train.json')
linemod_ann_sample = linemod_dataset.annotations[0]
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
corner_3d = linemod_ann_sample.corner_3d
# K = linemod_ann_sample.K
K = np.array([517.799858, 0.000000, 303.876287, 0.000000, 514.807834, 238.157119, 0.000000, 0.000000, 1.000000]).reshape(3,3)
distortion = np.array([0.001647, -0.105636, -0.002094, -0.006446, 0.000000])
linemod_image_sample = linemod_dataset.images[0]
dsize = (linemod_image_sample.width, linemod_image_sample.height)

infer_tests_pvnet(
    weight_path=weight_paths,
    model_name=model_names,
    dataset=datasets,
    test_name=test_names,
    kpt_3d=kpt_3d,
    corner_3d=corner_3d,
    K=K,
    distortion=distortion,
    dsize=dsize,
    show_pbar=True,
    blackout=True,
    show_preview=False,
    data_dump_dir='infer_data_dump',
    # video_dump_dir='video_dump',
    # img_dump_dir='img_dump',
    skip_if_data_dump_exists=True
)