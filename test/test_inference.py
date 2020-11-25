from clean_pvnet.infer.pvnet_inferer import PVNetInferer, PVNetFrameResultList
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.path_utils import recursively_get_all_filepaths_of_extension, get_rootname_from_path, get_dirpath_from_filepath
from common_utils.file_utils import dir_exists, file_exists, make_dir_if_not_exists
from tqdm import tqdm

inferer = PVNetInferer(
    # weight_path='/home/clayton/workspace/git/clean-pvnet/data/model/pvnet/custom/99.pth',
    # weight_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_weights/194.pth',
    weight_path='/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/499.pth'
)
model_name = '20201119_epoch499'

# img_dir = '/home/clayton/workspace/git/pvnet-rendering/test/renders1'
img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/nihonbashi2/organized'
linemod_dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')
linemod_ann_sample = linemod_dataset.annotations[0]
kpt_3d = linemod_ann_sample.fps_3d.copy()
kpt_3d.append(linemod_ann_sample.center_3d)
corner_3d = linemod_ann_sample.corner_3d
K = linemod_ann_sample.K
linemod_image_sample = linemod_dataset.images[0]
dsize = (linemod_image_sample.width, linemod_image_sample.height)

robot_camera_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201017_robot_camera'
infer_data_dump = f'{robot_camera_dir}/infer_dump'
make_dir_if_not_exists(infer_data_dump)

csv_paths = recursively_get_all_filepaths_of_extension(robot_camera_dir, extension='csv')
pbar = tqdm(total=len(csv_paths), unit='dataset(s)')
pbar.set_description('Inferring Test Datasets')
frame_result_list = PVNetFrameResultList()
for csv_path in csv_paths:
    test_name = get_rootname_from_path(csv_path)
    img_dir = f'{get_dirpath_from_filepath(csv_path)}/images'
    assert dir_exists(img_dir), f"Couldn't find image directory: {img_dir}"
    ann_path = f'{img_dir}/output.json'
    if not file_exists(ann_path):
        pbar.update()
        continue
    dataset = COCO_Dataset.load_from_path(ann_path, img_dir=img_dir)

    frame_result_list0 = inferer.infer_coco_dataset(
        dataset=dataset,
        kpt_3d=kpt_3d,
        corner_3d=corner_3d,
        K=K,
        blackout=True,
        dsize=dsize,
        show_preview=True,
        test_name=test_name,
        model_name=model_name,
        pred_dump_path='pred_dump.json'
    )
    frame_result_list += frame_result_list0
    pbar.update()
frame_result_list.save_to_path(f'{infer_data_dump}/{model_name}_infer.json', overwrite=True)
pbar.close()