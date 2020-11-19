# Assuming pvnet

from clean_pvnet.networks.pvnet.resnet18 import get_res_pvnet
from clean_pvnet.util.net_utils import custom_load_network
from clean_pvnet.util.pvnet_data_utils import compute_vertex
from clean_pvnet.datasets.transforms import make_transforms
from clean_pvnet.util import pvnet_pose_utils
import numpy as np
import torch
from PIL import Image
from common_utils.path_utils import get_filename
from annotation_utils.linemod.objects import Linemod_Dataset
import cv2

def get_network(vote_dim: int=18, seg_dim: int=2):
    return get_res_pvnet(vote_dim, seg_dim)

def make_network():
    return get_network()

def draw_corners(img: np.ndarray, corner_2d: np.ndarray, color: tuple=(255, 0, 0), thickness: int=2) -> np.ndarray:
    result = img.copy()
    indexes = [
        (0, 1), (1, 3), (3, 2), (2, 0), (0, 4), (4, 6), (6, 2),
        (5, 4), (4, 6), (6, 7), (7, 5), (5, 1), (1, 3), (3, 7)
    ]
    corner_data = corner_2d.tolist()
    for start_idx, end_idx in indexes:
        result = cv2.line(
            img=result,
            pt1=tuple([int(val) for val in corner_data[start_idx]]), pt2=tuple([int(val) for val in corner_data[end_idx]]),
            color=color, thickness=thickness
        )
    return result

def draw_pts2d(img: np.ndarray, pts2d: np.ndarray, color: tuple=(255, 0, 0), radius: int=2) -> np.ndarray:
    result = img.copy()
    for x, y in pts2d.tolist():
        result = cv2.circle(
            img=result, center=(int(x), int(y)),
            radius=radius, color=color, thickness=-1
        )
    return result

network = make_network().cuda()
epoch = custom_load_network(
    net=network,
    # weight_path='/home/clayton/workspace/git/clean-pvnet/data/model/pvnet/custom/14.pth',
    # weight_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_weights/194.pth',
    weight_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/linemod_weights/cat199.pth',
    resume=False,
    strict=True
)
network.eval()
print(network)

# Refer to clean-pvnet/lib/datasets/linemod/pvnet.py for an idea of what the input should be.

# img_dir = '/home/clayton/workspace/git/pvnet-rendering/test/renders1'
# dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')
img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/new/LINEMOD/cat/JPEGImages'
dataset = Linemod_Dataset.load_from_path(f'{img_dir}/../train.json')
transforms = make_transforms(is_train=False)
for linemod_image in dataset.images:
    # img_path = f'{img_dir}/{linemod_image.file_name}'
    img_path = f'{img_dir}/{get_filename(linemod_image.file_name)}'
    img = Image.open(img_path)
    orig_img = np.asarray(img)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    img = transforms.on_image(img)
    img = img.reshape(tuple([1] + list(img.shape)))
    img = torch.from_numpy(img)
    img = img.cuda()
    output = network(img)
    seg = output['seg']
    print(f'seg:\n{seg}')
    vertex = output['vertex']
    print(f'vertex:\n{vertex}')
    mask = output['mask']
    print(f'mask:\n{mask}')
    kpt_2d = output['kpt_2d'].detach().cpu().numpy()[0]
    print(f'kpt_2d:\n{kpt_2d}')

    # Visualize
    gt_ann = dataset.annotations.get(image_id=linemod_image.id)[0]
    result = orig_img.copy()

    gt_kpt_3d = gt_ann.fps_3d.to_list()
    gt_kpt_3d.append(gt_ann.center_3d.to_list())
    gt_kpt_3d = np.array(gt_kpt_3d)
    print(f'gt_kpt_3d.shape: {gt_kpt_3d.shape}, kpt_2d.shape: {kpt_2d.shape}')
    pose_pred = pvnet_pose_utils.pnp(gt_kpt_3d, kpt_2d, gt_ann.K.to_matrix())
    print(f'pose_pred:\n{pose_pred}')
    print(f'gt_ann.pose:\n{np.array(gt_ann.pose.to_list())}')
    corner_2d_gt = pvnet_pose_utils.project(gt_ann.corner_3d.to_numpy(), gt_ann.K.to_matrix(), np.array(gt_ann.pose.to_list()))
    corner_2d_pred = pvnet_pose_utils.project(gt_ann.corner_3d.to_numpy(), gt_ann.K.to_matrix(), pose_pred)
    print(f'corner_2d_gt:\n{corner_2d_gt}')
    print(f'corner_2d_pred:\n{corner_2d_pred}')

    from common_utils.cv_drawing_utils import cv_simple_image_viewer
    result = draw_pts2d(img=result, pts2d=gt_ann.fps_2d.to_numpy(), color=(0,255,0), radius=3)
    result = draw_corners(img=result, corner_2d=corner_2d_gt, color=(0,255,0), thickness=2)
    result = draw_pts2d(img=result, pts2d=kpt_2d, color=(0,0,255), radius=2)
    result = draw_corners(img=result, corner_2d=corner_2d_pred, color=(0,0,255), thickness=2)
    quit_flag = cv_simple_image_viewer(img=result, preview_width=1000)
    if quit_flag:
        break

    # anns = dataset.annotations.get(image_id=linemod_image.id)
    # for ann in anns:
    #     kpt_2d = np.concatenate([ann.fps_2d.to_numpy(), ann.center_2d.to_numpy()], axis=0)
    #     mask = (np.asarray(Image.open(ann.mask_path))).astype(np.uint8)
    #     vertex = compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
    #     input_dict = ret = {'inp': img, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': ann.image_id, 'meta': {}}
