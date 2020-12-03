import numpy as np
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from PIL.JpegImagePlugin import JpegImageFile
from PIL import Image

from common_utils.path_utils import get_filename
from common_utils.common_types.point import Point3D_List, Point2D_List, Point3D
from common_utils.common_types.angle import QuaternionList
from common_utils.common_types.bbox import BBox
from streamer.cv_viewer import cv_simple_image_viewer
from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset

from model_based_angle.pnp import PnP_Model

from ..networks.pvnet.resnet18 import get_res_pvnet
from ..util.net_utils import custom_load_network
from ..datasets.transforms import make_transforms
from ..util import pvnet_pose_utils
from ..util.draw_utils import draw_corners, draw_pts2d
from ..util.error_utils import get_type_error_message
from ..util.stream_writer import StreamWriter

def do_pnp(
    kpt_2d: np.ndarray, gt_kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
    camera_translation: np.ndarray=None, camera_quaternion: np.ndarray=None,
    distortion: np.ndarray=None,
    line_start_point3d: np.ndarray=None, line_end_point_3d: np.ndarray=None,
    units_per_meter: float=1.0
) -> (np.ndarray, np.ndarray, tuple):
    if line_start_point3d is None or line_end_point_3d is None:
        corner_3d_np = corner_3d.to_numpy() if isinstance(corner_3d, Point3D_List) else corner_3d
        front_center = corner_3d_np[[4, 5, 6, 7]].mean(axis=0)
        back_center = corner_3d_np[[0, 1, 2, 3]].mean(axis=0)
        direction_center = front_center + (front_center - back_center)

    pnp_model = PnP_Model(
        points_3d=gt_kpt_3d,
        camera_translation=np.array([0,0,0]) if camera_translation is None else camera_translation,
        camera_quaternion=np.array([0,0,0,1]) if camera_quaternion is None else camera_quaternion,
        distortion=distortion,
        camera_matrix=K,
        line_start_point_3d=front_center if line_start_point3d is None else line_start_point3d,
        line_end_point_3d=direction_center if line_end_point_3d is None else line_end_point_3d,
        units_per_meter=units_per_meter
    )
    kpt_2d_np = kpt_2d.to_numpy() if isinstance(kpt_2d, Point2D_List) else kpt_2d
    if len(kpt_2d_np[~np.all(kpt_2d_np == 0, axis=1)]) >= 6:
        object_world_position, object_world_rotation, (p1, p2) = pnp_model.calculate_pos_and_angle(kpt_2d=kpt_2d)
    else:
        object_world_position, object_world_rotation, (p1, p2) = None, None, (None, None)

    return object_world_position, object_world_rotation, (p1, p2)

def calc_pose_pred(kpt_2d: np.ndarray, gt_kpt_3d: np.ndarray, K: np.ndarray, func_name: str='calc_pose_pred') -> np.ndarray:
    if isinstance(kpt_2d, np.ndarray):
        kpt_2d0 = kpt_2d.copy()
    elif isinstance(kpt_2d, Point2D_List):
        kpt_2d0 = kpt_2d.to_numpy(demarcation=True)
    else:
        raise TypeError(
            get_type_error_message(
                func_name=func_name,
                acceptable_types=[np.ndarray, Point2D_List],
                unacceptable_type=type(kpt_2d),
                param_name='kpt_2d'
            )
        )
    if isinstance(gt_kpt_3d, np.ndarray):
        gt_kpt_3d0 = gt_kpt_3d.copy()
    elif isinstance(gt_kpt_3d, Point3D_List):
        gt_kpt_3d0 = gt_kpt_3d.to_numpy(demarcation=True)
    else:
        raise TypeError(
            get_type_error_message(
                func_name=func_name,
                acceptable_types=[np.ndarray, Point3D_List],
                unacceptable_type=type(gt_kpt_3d),
                param_name='gt_kpt_3d'
            )
        )
    if isinstance(K, np.ndarray):
        K0 = K.copy()
    elif isinstance(K, LinemodCamera):
        K0 = K.to_matrix()
    else:
        raise TypeError(
            get_type_error_message(
                func_name=func_name,
                acceptable_types=[np.ndarray, LinemodCamera],
                unacceptable_type=type(K),
                param_name='K'
            )
        )
    return pvnet_pose_utils.pnp(gt_kpt_3d0, kpt_2d0, K0)

def calc_corner_2d_pred(
    gt_corner_3d: np.ndarray, K: np.ndarray, pose_pred: np.ndarray,
    func_name: str='calc_corner_2d_pred'
):
    if isinstance(gt_corner_3d, np.ndarray):
        gt_corner_3d0 = gt_corner_3d.copy()
    elif isinstance(gt_corner_3d, Point3D_List):
        gt_corner_3d0 = gt_corner_3d.to_numpy(demarcation=True)
    else:
        raise TypeError(
            get_type_error_message(
                func_name=func_name,
                acceptable_types=[np.ndarray, Point3D_List],
                unacceptable_type=type(gt_corner_3d),
                param_name='gt_corner_3d'
            )
        )
    if isinstance(K, np.ndarray):
        K0 = K.copy()
    elif isinstance(K, LinemodCamera):
        K0 = K.to_matrix()
    else:
        raise TypeError(
            get_type_error_message(
                func_name=func_name,
                acceptable_types=[np.ndarray, LinemodCamera],
                unacceptable_type=type(K),
                param_name='K'
            )
        )
    if isinstance(pose_pred, np.ndarray):
        pose_pred0 = pose_pred.copy()
    elif isinstance(pose_pred, QuaternionList):
        pose_pred0 = pose_pred.to_numpy()
    else:
        raise TypeError(
            get_type_error_message(
                func_name=func_name,
                acceptable_types=[np.ndarray, QuaternionList],
                unacceptable_type=type(pose_pred),
                param_name='pose_pred'
            )
        )
    return pvnet_pose_utils.project(gt_corner_3d0, K0, pose_pred0)

class PVNetPrediction(BasicLoadableObject['PVNetPrediction']):
    def __init__(self, seg: np.ndarray, vertex: np.ndarray, mask: np.ndarray, kpt_2d: np.ndarray):
        self.seg = seg
        self.vertex = vertex
        self.mask = mask
        self.kpt_2d = kpt_2d
    
    def to_dict(self) -> dict:
        return {
            'seg': self.seg.tolist(),
            'vertex': self.vertex.tolist(),
            'mask': self.mask.tolist(),
            'kpt_2d': self.kpt_2d.tolist()
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        return PVNetPrediction(
            seg=np.array(item_dict['seg']),
            vertex=np.array(item_dict['vertex']),
            mask=np.array(item_dict['mask']),
            kpt_2d=np.array(item_dict['kpt_2d'])
        )

    def to_pose_pred(self, gt_kpt_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        return calc_pose_pred(
            kpt_2d=self.kpt_2d, gt_kpt_3d=gt_kpt_3d, K=K,
            func_name=f'{type(self).__name__}.to_pose_pred'
        )

    def to_corner_2d_pred(self, gt_corner_3d: np.ndarray, K: np.ndarray, pose_pred: np.ndarray):
        return calc_corner_2d_pred(
            gt_corner_3d=gt_corner_3d, K=K, pose_pred=pose_pred,
            func_name=f'{type(self).__name__}.to_corner_2d_pred'
        )

    def to_corner_2d(
        self, gt_kpt_3d: np.ndarray, gt_corner_3d: np.ndarray, K: np.ndarray
    ) -> Point3D_List:
        pose_pred = self.to_pose_pred(gt_kpt_3d=gt_kpt_3d, K=K)
        corner_2d_pred = self.to_corner_2d_pred(
            gt_corner_3d=gt_corner_3d, K=K, pose_pred=pose_pred
        )
        return Point2D_List.from_list(corner_2d_pred, demarcation=True)

    def to_pnp_pred(
        self, gt_kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
        camera_translation: np.ndarray=None, camera_quaternion: np.ndarray=None,
        distortion: np.ndarray=None,
        line_start_point3d: np.ndarray=None, line_end_point_3d: np.ndarray=None,
        units_per_meter: float=1.0
    ):
        pose_pred = self.to_pose_pred(gt_kpt_3d=gt_kpt_3d, K=K)
        corner_2d_pred = self.to_corner_2d_pred(
            gt_corner_3d=corner_3d, K=K, pose_pred=pose_pred
        )

        object_world_position, object_world_rotation, (p1, p2) = do_pnp(
            kpt_2d=self.kpt_2d, gt_kpt_3d=gt_kpt_3d, corner_3d=corner_3d, K=K,
            camera_translation=camera_translation, camera_quaternion=camera_quaternion,
            distortion=distortion,
            line_start_point3d=line_start_point3d, line_end_point_3d=line_end_point_3d,
            units_per_meter=units_per_meter
        )

        return PnpPrediction(
            kpt_2d=self.kpt_2d,
            pose=pose_pred,
            corner_2d=corner_2d_pred,
            object_world_position=object_world_position,
            object_world_rotation=object_world_rotation,
            p1=p1, p2=p2
        )

    def draw_pred(
        self, img: np.ndarray, corner_2d_pred: np.ndarray,
        color: tuple=(255,0,0), pt_radius: int=2, line_thickness: int=2
    ) -> np.ndarray:
        result = draw_pts2d(
            img=img, pts2d=self.kpt_2d,
            color=color, radius=pt_radius
        )
        result = draw_corners(
            img=result, corner_2d=corner_2d_pred,
            color=color, thickness=line_thickness
        )
        return result

    def draw(
        self, img: np.ndarray,
        gt_kpt_3d: np.ndarray, gt_corner_3d: np.ndarray, K: np.ndarray,
        color: tuple=(255,0,0), pt_radius: int=2, line_thickness: int=2
    ) -> np.ndarray:
        corner_2d_pred = self.to_corner_2d(
            gt_kpt_3d=gt_kpt_3d, gt_corner_3d=gt_corner_3d, K=K
        )
        return self.draw_pred(
            img=img, corner_2d_pred=corner_2d_pred.to_numpy(),
            color=color, pt_radius=pt_radius, line_thickness=line_thickness
        )

class PVNetPredictionList(
    BasicLoadableHandler['PVNetPredictionList', 'PVNetPrediction'],
    BasicHandler['PVNetPredictionList', 'PVNetPrediction']
):
    def __init__(self, pred_list: List[PVNetPrediction]=None):
        super().__init__(obj_type=PVNetPrediction, obj_list=pred_list)
        self.pred_list = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]):
        return PVNetPredictionList([PVNetPrediction.from_dict(item_dict) for item_dict in dict_list])

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.to_dict_list())

    def to_pnp_pred(
        self, gt_kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
        camera_translation: np.ndarray=None, camera_quaternion: np.ndarray=None,
        distortion: np.ndarray=None,
        line_start_point3d: np.ndarray=None, line_end_point_3d: np.ndarray=None,
        units_per_meter: float=1.0
    ):
        pnp_pred_list = PnpPredictionList()
        for pred in self:
            pnp_pred_list.append(
                pred.to_pnp_pred(
                    gt_kpt_3d=gt_kpt_3d,
                    corner_3d=corner_3d,
                    K=K,
                    camera_translation=camera_translation,
                    camera_quaternion=camera_quaternion,
                    distortion=distortion,
                    line_start_point3d=line_start_point3d,
                    line_end_point_3d=line_end_point_3d,
                    units_per_meter=units_per_meter
                )
            )
        return pnp_pred_list

    def draw(
        self, img: np.ndarray,
        gt_kpt_3d: np.ndarray, gt_corner_3d: np.ndarray, K: np.ndarray,
        color: tuple=(255,0,0), pt_radius: int=2, line_thickness: int=2
    ) -> np.ndarray:
        result = img.copy()
        for pred in self:
            result = pred.draw(
                img=result,
                gt_kpt_3d=gt_kpt_3d, gt_corner_3d=gt_corner_3d, K=K,
                color=color, pt_radius=pt_radius, line_thickness=line_thickness
            )
        return result

class PnpPrediction(BasicLoadableObject['PnpPrediction']):
    def __init__(
        self, kpt_2d: np.ndarray, pose: np.ndarray, corner_2d: np.ndarray,
        object_world_position: np.ndarray, object_world_rotation: np.ndarray,
        p1: tuple, p2: tuple
    ):
        super().__init__()
        self.kpt_2d = kpt_2d
        self.pose = pose
        self.corner_2d = corner_2d
        self.object_world_position = object_world_position
        self.object_world_rotation = object_world_rotation
        self.p1 = p1
        self.p2 = p2
    
    @property
    def position(self) -> Point3D:
        return Point3D.from_list(self.object_world_position.tolist())
    
    @property
    def distance_from_camera(self) -> float:
        return self.position.distance(Point3D.origin())

    def to_dict(self) -> dict:
        return {
            'kpt_2d': self.kpt_2d.tolist() if self.kpt_2d is not None else None,
            'pose': self.pose.tolist() if self.pose is not None else None,
            'corner_2d': self.corner_2d.tolist() if self.corner_2d is not None else None,
            'object_world_position': self.object_world_position.tolist() if self.object_world_position is not None else None,
            'object_world_rotation': self.object_world_rotation.tolist() if self.object_world_rotation is not None else None,
            'p1': self.p1,
            'p2': self.p2
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        return PnpPrediction(
            kpt_2d=np.array(item_dict['kpt_2d']),
            pose=np.array(item_dict['pose']),
            corner_2d=np.array(item_dict['corner_2d']),
            object_world_position=np.array(item_dict['object_world_position']),
            object_world_rotation=np.array(item_dict['object_world_rotation']),
            p1=item_dict['p1'],
            p2=item_dict['p2']
        )

    def to_pose_pred(self, gt_kpt_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        return calc_pose_pred(
            kpt_2d=self.kpt_2d, gt_kpt_3d=gt_kpt_3d, K=K,
            func_name=f'{type(self).__name__}.to_pose_pred'
        )

    def to_corner_2d_pred(self, gt_corner_3d: np.ndarray, K: np.ndarray, pose_pred: np.ndarray):
        return calc_corner_2d_pred(
            gt_corner_3d=gt_corner_3d, K=K, pose_pred=pose_pred,
            func_name=f'{type(self).__name__}.to_corner_2d_pred'
        )

    @classmethod
    def from_kpt_2d(
        cls, kpt_2d: np.ndarray, gt_kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
        camera_translation: np.ndarray=None, camera_quaternion: np.ndarray=None,
        distortion: np.ndarray=None,
        line_start_point3d: np.ndarray=None, line_end_point_3d: np.ndarray=None,
        units_per_meter: float=1.0
    ):
        pose_pred = calc_pose_pred(
            kpt_2d=kpt_2d, gt_kpt_3d=gt_kpt_3d, K=K
        )
        corner_2d_pred = calc_corner_2d_pred(
            gt_corner_3d=corner_3d, K=K, pose_pred=pose_pred
        )

        object_world_position, object_world_rotation, (p1, p2) = do_pnp(
            kpt_2d=kpt_2d, gt_kpt_3d=gt_kpt_3d, corner_3d=corner_3d, K=K,
            camera_translation=camera_translation, camera_quaternion=camera_quaternion,
            distortion=distortion,
            line_start_point3d=line_start_point3d, line_end_point_3d=line_end_point_3d,
            units_per_meter=units_per_meter
        )
        return PnpPrediction(
            kpt_2d=kpt_2d, pose=pose_pred, corner_2d=corner_2d_pred,
            object_world_position=object_world_position, object_world_rotation=object_world_rotation,
            p1=p1, p2=p2
        )

    def recalculate(
        self, gt_kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
        camera_translation: np.ndarray=None, camera_quaternion: np.ndarray=None,
        distortion: np.ndarray=None,
        line_start_point3d: np.ndarray=None, line_end_point_3d: np.ndarray=None,
        units_per_meter: float=1.0
    ):
        new = PnpPrediction.from_kpt_2d(
            kpt_2d=self.kpt_2d,
            gt_kpt_3d=gt_kpt_3d,
            corner_3d=corner_3d,
            K=K,
            distortion=distortion,
            camera_translation=camera_translation,
            camera_quaternion=camera_quaternion,
            line_start_point3d=line_start_point3d,
            line_end_point_3d=line_end_point_3d,
            units_per_meter=units_per_meter
        )
        self.pose = new.pose
        self.corner_2d = new.corner_2d
        self.object_world_position = new.object_world_position
        self.object_world_rotation = new.object_world_rotation
        self.p1 = new.p1
        self.p2 = new.p2

    def draw(
        self, img: np.ndarray,
        corners_line_color: tuple=(255,0,0),
        corners_line_thickness: int=2,
        corners_pt_radius: int=2,
        direction_line_color: tuple=(255,0,0),
        direction_line_thickness: int=5,
        direction_point_color: tuple=(0,255,0)
    ) -> np.ndarray:
        result = draw_pts2d(
            img=img, pts2d=self.kpt_2d,
            color=corners_line_color, radius=corners_pt_radius
        )
        result = draw_corners(
            img=result, corner_2d=self.corner_2d,
            color=corners_line_color, thickness=corners_line_thickness
        )
        if self.p1 is not None and self.p2 is not None:
            result = PnP_Model.draw_line(
                img=result, p1=self.p1, p2=self.p2,
                line_color=direction_line_color, point_color=direction_point_color,
                thickness=direction_line_thickness
            )
        return result

class PnpPredictionList(
    BasicLoadableHandler['PnpPredictionList', 'PnpPrediction'],
    BasicHandler['PnpPredictionList', 'PnpPrediction']
):
    def __init__(self, pred_list: List[PnpPrediction]=None):
        super().__init__(obj_type=PnpPrediction, obj_list=pred_list)
        self.pred_list = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]):
        return PnpPredictionList([PnpPrediction.from_dict(item_dict) for item_dict in dict_list])

    def draw(
        self, img: np.ndarray,
        corners_line_color: tuple=(0,0,255),
        corners_line_thickness: int=2,
        corners_pt_radius: int=2,
        direction_line_color: tuple=(255,0,0),
        direction_line_thickness: int=5,
        direction_point_color: tuple=(0,255,0)
    ) -> np.ndarray:
        result = img.copy()
        for pred in self:
            result = pred.draw(
                img=result,
                corners_line_color=corners_line_color,
                corners_pt_radius=corners_pt_radius,
                corners_line_thickness=corners_line_thickness,
                direction_line_color=direction_line_color,
                direction_line_thickness=direction_line_thickness,
                direction_point_color=direction_point_color
            )
        return result

class PVNetFrameResult(BasicLoadableObject['PVNetFrameResult']):
    def __init__(self, frame: str, pred_list: PnpPredictionList=None, test_name: str=None, model_name: str=None):
        super().__init__()
        self.frame = frame
        self.pred_list = pred_list if pred_list is not None else PnpPredictionList()
        self.test_name = test_name
        self.model_name = model_name

    @classmethod
    def from_dict(cls, item_dict: dict):
        return PVNetFrameResult(
            frame=item_dict['frame'],
            pred_list=PnpPredictionList.from_dict_list(item_dict['pred_list']),
            test_name=item_dict['test_name'],
            model_name=item_dict['model_name']
        )

class PVNetFrameResultList(
    BasicLoadableHandler['PVNetFrameResultList', 'PVNetFrameResult'],
    BasicHandler['PVNetFrameResultList', 'PVNetFrameResult']
):
    def __init__(self, result_list: List[PVNetFrameResult]=None):
        super().__init__(obj_type=PVNetFrameResult, obj_list=result_list)
        self.result_list = self.obj_list
    
    @classmethod
    def from_dict_list(cls, dict_list: List[dict]):
        return PVNetFrameResultList([PVNetFrameResult.from_dict(item_dict) for item_dict in dict_list])

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.to_dict_list())
    
class PVNetInferer:
    def __init__(self, weight_path: str, vote_dim: int=18, seg_dim: int=2):
        self.network = get_res_pvnet(vote_dim, seg_dim).cuda()
        custom_load_network(
            net=self.network,
            weight_path=weight_path,
            resume=True,
            strict=True
        )
        self.network.eval()
        self.transforms = make_transforms(is_train=False)
    
    def _predict(self, img: np.ndarray, bbox: BBox=None) -> Dict[str, torch.cuda.FloatTensor]:
        # Convert to JpegImageFile
        if isinstance(img, JpegImageFile):
            img0 = img.copy()
        elif isinstance(img, np.ndarray):
            img0 = img.copy()
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img0 = Image.fromarray(img0)
        else:
            acceptable_types = ""
            for i, acceptable_type in enumerate([np.ndarray, JpegImageFile]):
                if i == 0:
                    acceptable_types += f'\t{acceptable_type.__name__}'
                else:
                    acceptable_types += f'\n\t{acceptable_type.__name__}'
            raise TypeError(
                get_type_error_message(
                    func_name='PVNetInferer._predict',
                    acceptable_types=[np.ndarray, JpegImageFile],
                    unacceptable_type=type(img),
                    param_name='img'
                )
            )
        
        # Blackout Region Outside BBox If Given BBox
        if bbox is not None:
            img0 = np.array(img0)
            img0 = bbox.crop_and_paste(src_img=img0, dst_img=np.zeros_like(img0))
            img0 = Image.fromarray(img0)

        # Infer
        img0 = self.transforms.on_image(img0)
        img0 = img0.reshape(tuple([1] + list(img0.shape)))
        img0 = torch.from_numpy(img0)
        img0 = img0.cuda()
        return self.network(img0)
    
    def predict(self, img: np.ndarray, bbox: BBox=None) -> PVNetPrediction:
        output = self._predict(img=img, bbox=bbox)
        return PVNetPrediction(
            seg=output['seg'].detach().cpu().numpy()[0],
            vertex=output['vertex'].detach().cpu().numpy()[0],
            mask=output['mask'].detach().cpu().numpy()[0],
            kpt_2d=output['kpt_2d'].detach().cpu().numpy()[0],
        )

    def infer_linemod_dataset(
        self, dataset: Linemod_Dataset, img_dir: str, blackout: bool=False, show_pbar: bool=True,
        show_preview: bool=False, video_save_path: str=None, dump_dir: str=None,
        accumulate_pred_dump: bool=True, pred_dump_path: str=None, test_name: str=None
    ) -> PVNetFrameResultList:
        stream_writer = StreamWriter(show_preview=show_preview, video_save_path=video_save_path, dump_dir=dump_dir)
        pbar = tqdm(total=len(dataset.images), unit='image(s)') if show_pbar else None
        frame_result_list = PVNetFrameResultList() if pred_dump_path is not None or accumulate_pred_dump else None
        for linemod_image in dataset.images:
            file_name = get_filename(linemod_image.file_name)
            if pbar is not None:
                pbar.set_description(file_name)
            img_path = f'{img_dir}/{file_name}'
            img = Image.open(img_path)

            orig_img = np.asarray(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            result = orig_img.copy()

            gt_ann = dataset.annotations.get(image_id=linemod_image.id)[0]
            if blackout:
                xmin = int(gt_ann.corner_2d.to_numpy(demarcation=True)[:, 0].min())
                xmax = int(gt_ann.corner_2d.to_numpy(demarcation=True)[:, 0].max())
                ymin = int(gt_ann.corner_2d.to_numpy(demarcation=True)[:, 1].min())
                ymax = int(gt_ann.corner_2d.to_numpy(demarcation=True)[:, 1].max())
                bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                bbox = bbox.clip_at_bounds(frame_shape=result.shape)
                result = bbox.crop_and_paste(src_img=result, dst_img=np.zeros_like(result))
            else:
                bbox = None
            pred = self.predict(img=img, bbox=bbox)
            if frame_result_list is not None:
                frame_result = PVNetFrameResult(frame=file_name, pred_list=PVNetPredictionList([pred]), test_name=test_name)
                frame_result_list.append(frame_result)

            gt_kpt_3d = gt_ann.fps_3d.copy()
            gt_kpt_3d.append(gt_ann.center_3d)
            pose_pred = pred.to_pose_pred(gt_kpt_3d=gt_kpt_3d, K=gt_ann.K)
            corner_2d_gt = pvnet_pose_utils.project(gt_ann.corner_3d.to_numpy(), gt_ann.K.to_matrix(), np.array(gt_ann.pose.to_list()))
            corner_2d_pred = pred.to_corner_2d_pred(gt_corner_3d=gt_ann.corner_3d, K=gt_ann.K, pose_pred=pose_pred)

            result = draw_pts2d(img=result, pts2d=gt_ann.fps_2d.to_numpy(), color=(0,255,0), radius=3)
            result = draw_corners(img=result, corner_2d=corner_2d_gt, color=(0,255,0), thickness=2)
            result = draw_pts2d(img=result, pts2d=pred.kpt_2d, color=(0,0,255), radius=2)
            result = draw_corners(img=result, corner_2d=corner_2d_pred, color=(0,0,255), thickness=2)
            stream_writer.step(img=result, file_name=file_name)
            if pbar is not None:
                pbar.update()
        stream_writer.close()
        if pred_dump_path is not None:
            frame_result_list.save_to_path(pred_dump_path, overwrite=True)
        pbar.close()
        return frame_result_list
    
    def infer_coco_dataset(
        self,
        dataset: COCO_Dataset,
        kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
        camera_translation: np.ndarray=None, camera_quaternion: np.ndarray=None,
        distortion: np.ndarray=None,
        line_start_point3d: np.ndarray=None, line_end_point_3d: np.ndarray=None,
        units_per_meter: float=1.0,
        blackout: bool=False, dsize: (int, int)=None, show_pbar: bool=True,
        show_preview: bool=False, video_save_path: str=None, dump_dir: str=None,
        accumulate_pred_dump: bool=True, pred_dump_path: str=None, test_name: str=None, model_name: str=None
    ) -> PVNetFrameResultList:
        stream_writer = StreamWriter(show_preview=show_preview, video_save_path=video_save_path, dump_dir=dump_dir)
        pbar = tqdm(total=len(dataset.images), unit='image(s)') if show_pbar else None
        frame_result_list = PVNetFrameResultList() if pred_dump_path is not None or accumulate_pred_dump else None
        for coco_image in dataset.images:
            file_name = get_filename(coco_image.file_name)
            if pbar is not None:
                pbar.set_description(file_name)
            img = Image.open(coco_image.coco_url)
            orig_img_w, orig_img_h = img.size

            if dsize is not None:
                img = img.resize(dsize)
                img_w, img_h = img.size
                xscale = img_w / orig_img_w
                yscale = img_h / orig_img_h

            orig_img = np.asarray(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

            anns = dataset.annotations.get_annotations_from_imgIds([coco_image.id])
            pred_list = PnpPredictionList()
            for ann in anns:
                working_img = orig_img.copy()
                if dsize is not None:
                    bbox = ann.bbox.resize(orig_frame_shape=[orig_img_h, orig_img_w, 3], new_frame_shape=[img_h, img_w, 3])
                else:
                    bbox = ann.bbox.copy()
                if blackout:
                    bbox = bbox.clip_at_bounds(frame_shape=working_img.shape)
                    working_img = bbox.crop_and_paste(src_img=working_img, dst_img=np.zeros_like(working_img))
                pred = self.predict(img=working_img, bbox=bbox if blackout else None)
                pred_list.append(pred.to_pnp_pred(
                    gt_kpt_3d=kpt_3d,
                    corner_3d=corner_3d,
                    K=K,
                    camera_translation=camera_translation,
                    camera_quaternion=camera_quaternion,
                    distortion=distortion,
                    line_start_point3d=line_start_point3d,
                    line_end_point_3d=line_end_point_3d,
                    units_per_meter=units_per_meter
                ))
            if frame_result_list is not None:
                frame_result = PVNetFrameResult(frame=file_name, pred_list=pred_list, test_name=test_name, model_name=model_name)
                frame_result_list.append(frame_result)
            result = pred_list.draw(img=orig_img)
            stream_writer.step(img=result, file_name=file_name)
            if pbar is not None:
                pbar.update()
        stream_writer.close()
        if pred_dump_path is not None:
            frame_result_list.save_to_path(pred_dump_path, overwrite=True)
        pbar.close()
        return frame_result_list