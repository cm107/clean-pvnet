from __future__ import annotations
import numpy as np
import torch
import cv2
from tqdm import tqdm
from typing import List, Dict
from PIL.JpegImagePlugin import JpegImageFile
from PIL import Image

from common_utils.path_utils import get_filename
from common_utils.common_types.point import Point3D_List, Point2D_List
from common_utils.common_types.angle import QuaternionList
from common_utils.common_types.bbox import BBox
from common_utils.cv_drawing_utils import cv_simple_image_viewer
from common_utils.base.basic import BasicLoadableObject, BasicLoadableHandler, BasicHandler
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset

from ..networks.pvnet.resnet18 import get_res_pvnet
from ..util.net_utils import custom_load_network
from ..datasets.transforms import make_transforms
from ..util import pvnet_pose_utils
from ..util.draw_utils import draw_corners, draw_pts2d
from ..util.error_utils import get_type_error_message
from ..util.stream_writer import StreamWriter

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
    def from_dict(cls, item_dict: dict) -> PVNetPrediction:
        return PVNetPrediction(
            seg=np.array(item_dict['seg']),
            vertex=np.array(item_dict['vertex']),
            mask=np.array(item_dict['mask']),
            kpt_2d=np.array(item_dict['kpt_2d'])
        )

    def to_pose_pred(self, gt_kpt_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        if isinstance(gt_kpt_3d, np.ndarray):
            gt_kpt_3d0 = gt_kpt_3d.copy()
        elif isinstance(gt_kpt_3d, Point3D_List):
            gt_kpt_3d0 = gt_kpt_3d.to_numpy(demarcation=True)
        else:
            raise TypeError(
                get_type_error_message(
                    func_name=f'{type(self).__name__}.to_pose_pred',
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
                    func_name=f'{type(self).__name__}.to_pose_pred',
                    acceptable_types=[np.ndarray, LinemodCamera],
                    unacceptable_type=type(K),
                    param_name='K'
                )
            )
        return pvnet_pose_utils.pnp(gt_kpt_3d0, self.kpt_2d, K0)

    def to_corner_2d_pred(self, gt_corner_3d: np.ndarray, K: np.ndarray, pose_pred: np.ndarray):
        if isinstance(gt_corner_3d, np.ndarray):
            gt_corner_3d0 = gt_corner_3d.copy()
        elif isinstance(gt_corner_3d, Point3D_List):
            gt_corner_3d0 = gt_corner_3d.to_numpy(demarcation=True)
        else:
            raise TypeError(
                get_type_error_message(
                    func_name=f'{type(self).__name__}.to_corner_2d_pred',
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
                    func_name=f'{type(self).__name__}.to_corner_2d_pred',
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
                    func_name=f'{type(self).__name__}.to_corner_2d_pred',
                    acceptable_types=[np.ndarray, QuaternionList],
                    unacceptable_type=type(pose_pred),
                    param_name='pose_pred'
                )
            )
        return pvnet_pose_utils.project(gt_corner_3d0, K0, pose_pred0)

    def to_corner_2d(
        self, gt_kpt_3d: np.ndarray, gt_corner_3d: np.ndarray, K: np.ndarray
    ) -> Point3D_List:
        pose_pred = self.to_pose_pred(gt_kpt_3d=gt_kpt_3d, K=K)
        corner_2d_pred = self.to_corner_2d_pred(
            gt_corner_3d=gt_corner_3d, K=K, pose_pred=pose_pred
        )
        return Point2D_List.from_list(corner_2d_pred, demarcation=True)
    
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
    def from_dict_list(cls, dict_list: List[dict]) -> PVNetPredictionList:
        return PVNetPredictionList([PVNetPrediction.from_dict(item_dict) for item_dict in dict_list])

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
    
    def _predict(self, img: np.ndarray, bbox: BBox=None) -> Dict[str, torch.cuda.Tensor]:
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
        show_preview: bool=False, video_save_path: str=None, dump_dir: str=None
    ):
        stream_writer = StreamWriter(show_preview=show_preview, video_save_path=video_save_path, dump_dir=dump_dir)
        pbar = tqdm(total=len(dataset.images), unit='image(s)') if show_pbar else None
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
        pbar.close()
    
    def infer_coco_dataset(
        self,
        dataset: COCO_Dataset,
        kpt_3d: np.ndarray, corner_3d: np.ndarray, K: np.ndarray,
        blackout: bool=False, dsize: (int, int)=None, show_pbar: bool=True,
        show_preview: bool=False, video_save_path: str=None, dump_dir: str=None
    ):
        stream_writer = StreamWriter(show_preview=show_preview, video_save_path=video_save_path, dump_dir=dump_dir)
        pbar = tqdm(total=len(dataset.images), unit='image(s)') if show_pbar else None
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
            pred_list = PVNetPredictionList()
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
                pred_list.append(pred)
            result = pred_list.draw(
                img=orig_img,
                gt_kpt_3d=kpt_3d, gt_corner_3d=corner_3d, K=K,
                color=(0,0,255), pt_radius=2, line_thickness=2
            )
            stream_writer.step(img=result, file_name=file_name)
            if pbar is not None:
                pbar.update()
        stream_writer.close()
        pbar.close()