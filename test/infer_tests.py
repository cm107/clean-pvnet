from common_utils.file_utils import file_exists, dir_exists
from common_utils.path_utils import recursively_get_all_filepaths_of_extension, \
    get_rootname_from_path, get_dirpath_from_filepath
from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.linemod.objects import Linemod_Dataset
from clean_pvnet.infer.pvnet_inferer import infer_tests_pvnet

weight_paths = [
    '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/199.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/49.pth',
    # '/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201209/19.pth'
]
model_names = [
    'pvnet20201209-epoch199',
    # 'pvnet20201209-epoch49',
    # 'pvnet20201209-epoch19'
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
K = linemod_ann_sample.K
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
    dsize=dsize,
    show_pbar=True,
    blackout=True,
    show_preview=True,
    data_dump_dir='infer_data_dump',
    video_dump_dir='video_dump',
    img_dump_dir='img_dump'
)