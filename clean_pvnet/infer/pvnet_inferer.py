from __future__ import annotations
import warnings
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
from common_utils.base.basic import (
    BasicLoadableObject,
    BasicLoadableHandler,
    BasicHandler,
)
from annotation_utils.linemod.objects import Linemod_Dataset, LinemodCamera
from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.coco.wrappers import infer_tests_wrapper
from streamer.recorder.stream_writer import StreamWriter

from model_based_angle.pnp import PnP_Model

from ..networks.pvnet.resnet18 import get_res_pvnet
from ..util.net_utils import custom_load_network
from ..datasets.transforms import make_transforms
from ..util import pvnet_pose_utils
from ..util.draw_utils import draw_corners, draw_pts2d
from ..util.error_utils import get_type_error_message

warnings.filterwarnings("ignore", category=UserWarning)


def do_pnp(
    kpt_2d: np.ndarray,
    gt_kpt_3d: np.ndarray,
    corner_3d: np.ndarray,
    K: np.ndarray,
    camera_translation: np.ndarray = None,
    camera_quaternion: np.ndarray = None,
    distortion: np.ndarray = None,
    line_start_point3d: np.ndarray = None,
    line_end_point_3d: np.ndarray = None,
    units_per_meter: float = 1.0,
) -> (np.ndarray, np.ndarray, tuple):
    if line_start_point3d is None or line_end_point_3d is None:
        corner_3d_np = (
            corner_3d.to_numpy() if isinstance(corner_3d, Point3D_List) else corner_3d
        )
        front_center = corner_3d_np[[4, 5, 6, 7]].mean(axis=0)
        back_center = corner_3d_np[[0, 1, 2, 3]].mean(axis=0)
        direction_center = front_center + (front_center - back_center)

    pnp_model = PnP_Model(
        points_3d=gt_kpt_3d,
        camera_translation=np.array([0, 0, 0])
        if camera_translation is None
        else camera_translation,
        camera_quaternion=np.array([0, 0, 0, 1])
        if camera_quaternion is None
        else camera_quaternion,
        distortion=distortion,
        camera_matrix=K,
        line_start_point_3d=front_center
        if line_start_point3d is None
        else line_start_point3d,
        line_end_point_3d=direction_center
        if line_end_point_3d is None
        else line_end_point_3d,
        units_per_meter=units_per_meter,
    )
    kpt_2d_np = kpt_2d.to_numpy() if isinstance(kpt_2d, Point2D_List) else kpt_2d
    if len(kpt_2d_np[~np.all(kpt_2d_np == 0, axis=1)]) >= 6:
        (
            object_world_position,
            object_world_rotation,
            (p1, p2),
        ) = pnp_model.calculate_pos_and_angle(kpt_2d=kpt_2d)
    else:
        object_world_position, object_world_rotation, (p1, p2) = (
            None,
            None,
            (None, None),
        )

    return object_world_position, object_world_rotation, (p1, p2)


def calc_pose_pred(
    kpt_2d: np.ndarray,
    gt_kpt_3d: np.ndarray,
    K: np.ndarray,
    func_name: str = "calc_pose_pred",
) -> np.ndarray:
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
                param_name="kpt_2d",
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
                param_name="gt_kpt_3d",
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
                param_name="K",
            )
        )
    return pvnet_pose_utils.pnp(gt_kpt_3d0, kpt_2d0, K0)


def calc_corner_2d_pred(
    gt_corner_3d: np.ndarray,
    K: np.ndarray,
    pose_pred: np.ndarray,
    func_name: str = "calc_corner_2d_pred",
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
                param_name="gt_corner_3d",
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
                param_name="K",
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
                param_name="pose_pred",
            )
        )
    return pvnet_pose_utils.project(gt_corner_3d0, K0, pose_pred0)


class PVNetPrediction(BasicLoadableObject["PVNetPrediction"]):
    def __init__(
        self, seg: np.ndarray, vertex: np.ndarray, mask: np.ndarray, kpt_2d: np.ndarray
    ):
        self.seg = seg
        self.vertex = vertex
        self.mask = mask
        self.kpt_2d = kpt_2d

    def to_dict(self) -> dict:
        return {
            "seg": self.seg.tolist(),
            "vertex": self.vertex.tolist(),
            "mask": self.mask.tolist(),
            "kpt_2d": self.kpt_2d.tolist(),
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> PVNetPrediction:
        return PVNetPrediction(
            seg=np.array(item_dict["seg"]),
            vertex=np.array(item_dict["vertex"]),
            mask=np.array(item_dict["mask"]),
            kpt_2d=np.array(item_dict["kpt_2d"]),
        )

    def to_pose_pred(self, gt_kpt_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        return calc_pose_pred(
            kpt_2d=self.kpt_2d,
            gt_kpt_3d=gt_kpt_3d,
            K=K,
            func_name=f"{type(self).__name__}.to_pose_pred",
        )

    def to_corner_2d_pred(
        self, gt_corner_3d: np.ndarray, K: np.ndarray, pose_pred: np.ndarray
    ):
        return calc_corner_2d_pred(
            gt_corner_3d=gt_corner_3d,
            K=K,
            pose_pred=pose_pred,
            func_name=f"{type(self).__name__}.to_corner_2d_pred",
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
        self,
        gt_kpt_3d: np.ndarray,
        corner_3d: np.ndarray,
        K: np.ndarray,
        camera_translation: np.ndarray = None,
        camera_quaternion: np.ndarray = None,
        distortion: np.ndarray = None,
        line_start_point3d: np.ndarray = None,
        line_end_point_3d: np.ndarray = None,
        units_per_meter: float = 1.0,
    ) -> PnpPrediction:
        pose_pred = self.to_pose_pred(gt_kpt_3d=gt_kpt_3d, K=K)
        corner_2d_pred = self.to_corner_2d_pred(
            gt_corner_3d=corner_3d, K=K, pose_pred=pose_pred
        )

        object_world_position, object_world_rotation, (p1, p2) = do_pnp(
            kpt_2d=self.kpt_2d,
            gt_kpt_3d=gt_kpt_3d,
            corner_3d=corner_3d,
            K=K,
            camera_translation=camera_translation,
            camera_quaternion=camera_quaternion,
            distortion=distortion,
            line_start_point3d=line_start_point3d,
            line_end_point_3d=line_end_point_3d,
            units_per_meter=units_per_meter,
        )

        return PnpPrediction(
            kpt_2d=self.kpt_2d,
            pose=pose_pred,
            corner_2d=corner_2d_pred,
            object_world_position=object_world_position,
            object_world_rotation=object_world_rotation,
            p1=p1,
            p2=p2,
        )

    def draw_pred(
        self,
        img: np.ndarray,
        corner_2d_pred: np.ndarray,
        color: tuple = (255, 0, 0),
        pt_radius: int = 2,
        line_thickness: int = 2,
    ) -> np.ndarray:
        result = draw_pts2d(img=img, pts2d=self.kpt_2d, color=color, radius=pt_radius)
        result = draw_corners(
            img=result, corner_2d=corner_2d_pred, color=color, thickness=line_thickness
        )
        return result

    def draw(
        self,
        img: np.ndarray,
        gt_kpt_3d: np.ndarray,
        gt_corner_3d: np.ndarray,
        K: np.ndarray,
        color: tuple = (255, 0, 0),
        pt_radius: int = 2,
        line_thickness: int = 2,
    ) -> np.ndarray:
        corner_2d_pred = self.to_corner_2d(
            gt_kpt_3d=gt_kpt_3d, gt_corner_3d=gt_corner_3d, K=K
        )
        return self.draw_pred(
            img=img,
            corner_2d_pred=corner_2d_pred.to_numpy(),
            color=color,
            pt_radius=pt_radius,
            line_thickness=line_thickness,
        )


class PVNetPredictionList(
    BasicLoadableHandler["PVNetPredictionList", "PVNetPrediction"],
    BasicHandler["PVNetPredictionList", "PVNetPrediction"],
):
    def __init__(self, pred_list: List[PVNetPrediction] = None):
        super().__init__(obj_type=PVNetPrediction, obj_list=pred_list)
        self.pred_list = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> PVNetPredictionList:
        return PVNetPredictionList(
            [PVNetPrediction.from_dict(item_dict) for item_dict in dict_list]
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.to_dict_list())

    def to_pnp_pred(
        self,
        gt_kpt_3d: np.ndarray,
        corner_3d: np.ndarray,
        K: np.ndarray,
        camera_translation: np.ndarray = None,
        camera_quaternion: np.ndarray = None,
        distortion: np.ndarray = None,
        line_start_point3d: np.ndarray = None,
        line_end_point_3d: np.ndarray = None,
        units_per_meter: float = 1.0,
    ) -> PnpPredictionList:
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
                    units_per_meter=units_per_meter,
                )
            )
        return pnp_pred_list

    def draw(
        self,
        img: np.ndarray,
        gt_kpt_3d: np.ndarray,
        gt_corner_3d: np.ndarray,
        K: np.ndarray,
        color: tuple = (255, 0, 0),
        pt_radius: int = 2,
        line_thickness: int = 2,
    ) -> np.ndarray:
        result = img.copy()
        for pred in self:
            result = pred.draw(
                img=result,
                gt_kpt_3d=gt_kpt_3d,
                gt_corner_3d=gt_corner_3d,
                K=K,
                color=color,
                pt_radius=pt_radius,
                line_thickness=line_thickness,
            )
        return result


class PnpDrawSettings(BasicLoadableObject["PnpDrawSettings"]):
    def __init__(
        self,
        corners_line_color: tuple = (0, 0, 255),
        corners_line_thickness: int = 2,
        corners_pt_radius: int = 2,
        direction_line_color: tuple = (255, 0, 0),
        direction_line_thickness: int = 5,
        direction_point_color: tuple = (0, 255, 0),
    ):
        """Drawing settings for Pnp result calculated from pvnet output.

        Args:
            corners_line_color (tuple, optional):
                The color of the lines that connect corner_3d.
                (This refers to the 3D box.)
                Defaults to (0,0,255).
            corners_line_thickness (int, optional):
                The thickness of the lines that connect corner_3d.
                (This refers to the 3D box.)
                Defaults to 2.
            corners_pt_radius (int, optional):
                The radius of the keypoints that were inputted into the
                Pnp model.
                (These are the points that were directly outputted
                from the pvnet model.)
                Defaults to 2.
            direction_line_color (tuple, optional):
                The color of the line indicating which way the object
                is facing.
                Defaults to (255,0,0).
            direction_line_thickness (int, optional):
                The thickness of the line indicating which way the object
                is facing.
                Defaults to 5.
            direction_point_color (tuple, optional):
                The color of the point at the end of the line indicating
                which way the
                object is facing.
                Defaults to (0,255,0).
        """
        self.corners_line_color = corners_line_color
        self.corners_line_thickness = corners_line_thickness
        self.corners_pt_radius = corners_pt_radius
        self.direction_line_color = direction_line_color
        self.direction_line_thickness = direction_line_thickness
        self.direction_point_color = direction_point_color


class PnpPrediction(BasicLoadableObject["PnpPrediction"]):
    def __init__(
        self,
        kpt_2d: np.ndarray,
        pose: np.ndarray,
        corner_2d: np.ndarray,
        object_world_position: np.ndarray,
        object_world_rotation: np.ndarray,
        p1: tuple,
        p2: tuple,
    ):
        """This class represents the data of a single Pnp calculation, which
        in turn corresponds to a single pvnet inference.

        Args:
            kpt_2d (np.ndarray):
                The 2D keypoints that were directly outputted from the pvnet
                model.
                These keypoints, together with other known values (i.e. camera
                matrix, etc.),
                are used to calculate the rest of the values shown below.
            pose (np.ndarray):
                Array that represents the pose of the object.
            corner_2d (np.ndarray):
                These points are a projection of corner_3d, the corners of the
                3D bounding box, to the camera plane. These values are used
                draw a visualization of the 3D bounding box.
            object_world_position (np.ndarray):
                3D position of the detected object in 3D space relative to
                the camera.
            object_world_rotation (np.ndarray):
                3D rotation of the detected object in 3D space relative to
                the camera.
            p1 (tuple):
                Starting point of directional line. This is used for drawing a
                visualization of the directional line.
            p2 (tuple):
                Ending point of directional line. This is used for drawing a
                visualization of the directional line.
        """
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
        return (
            Point3D.from_list(self.object_world_position.tolist())
            if self.object_world_position is not None
            else None
        )

    @property
    def distance_from_camera(self) -> float:
        return (
            self.position.distance(Point3D.origin())
            if self.position is not None
            else None
        )

    def to_dict(self) -> dict:
        return {
            "kpt_2d": self.kpt_2d.tolist()
            if self.kpt_2d is not None else None,
            "pose": self.pose.tolist() if self.pose is not None else None,
            "corner_2d": self.corner_2d.tolist()
            if self.corner_2d is not None
            else None,
            "object_world_position": self.object_world_position.tolist()
            if self.object_world_position is not None
            else None,
            "object_world_rotation": self.object_world_rotation.tolist()
            if self.object_world_rotation is not None
            else None,
            "p1": self.p1,
            "p2": self.p2,
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> PnpPrediction:
        return PnpPrediction(
            kpt_2d=np.array(item_dict["kpt_2d"]),
            pose=np.array(item_dict["pose"]),
            corner_2d=np.array(item_dict["corner_2d"]),
            object_world_position=np.array(item_dict["object_world_position"]),
            object_world_rotation=np.array(item_dict["object_world_rotation"]),
            p1=item_dict["p1"],
            p2=item_dict["p2"],
        )

    def to_pose_pred(self, gt_kpt_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        return calc_pose_pred(
            kpt_2d=self.kpt_2d,
            gt_kpt_3d=gt_kpt_3d,
            K=K,
            func_name=f"{type(self).__name__}.to_pose_pred",
        )

    def to_corner_2d_pred(
        self, gt_corner_3d: np.ndarray, K: np.ndarray, pose_pred: np.ndarray
    ):
        return calc_corner_2d_pred(
            gt_corner_3d=gt_corner_3d,
            K=K,
            pose_pred=pose_pred,
            func_name=f"{type(self).__name__}.to_corner_2d_pred",
        )

    @classmethod
    def from_kpt_2d(
        cls,
        kpt_2d: np.ndarray,
        gt_kpt_3d: np.ndarray,
        corner_3d: np.ndarray,
        K: np.ndarray,
        camera_translation: np.ndarray = None,
        camera_quaternion: np.ndarray = None,
        distortion: np.ndarray = None,
        line_start_point3d: np.ndarray = None,
        line_end_point_3d: np.ndarray = None,
        units_per_meter: float = 1.0,
    ) -> PnpPrediction:
        pose_pred = calc_pose_pred(kpt_2d=kpt_2d, gt_kpt_3d=gt_kpt_3d, K=K)
        corner_2d_pred = calc_corner_2d_pred(
            gt_corner_3d=corner_3d, K=K, pose_pred=pose_pred
        )

        object_world_position, object_world_rotation, (p1, p2) = do_pnp(
            kpt_2d=kpt_2d,
            gt_kpt_3d=gt_kpt_3d,
            corner_3d=corner_3d,
            K=K,
            camera_translation=camera_translation,
            camera_quaternion=camera_quaternion,
            distortion=distortion,
            line_start_point3d=line_start_point3d,
            line_end_point_3d=line_end_point_3d,
            units_per_meter=units_per_meter,
        )
        return PnpPrediction(
            kpt_2d=kpt_2d,
            pose=pose_pred,
            corner_2d=corner_2d_pred,
            object_world_position=object_world_position,
            object_world_rotation=object_world_rotation,
            p1=p1,
            p2=p2,
        )

    def recalculate(
        self,
        gt_kpt_3d: np.ndarray,
        corner_3d: np.ndarray,
        K: np.ndarray,
        camera_translation: np.ndarray = None,
        camera_quaternion: np.ndarray = None,
        distortion: np.ndarray = None,
        line_start_point3d: np.ndarray = None,
        line_end_point_3d: np.ndarray = None,
        units_per_meter: float = 1.0,
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
            units_per_meter=units_per_meter,
        )
        self.pose = new.pose
        self.corner_2d = new.corner_2d
        self.object_world_position = new.object_world_position
        self.object_world_rotation = new.object_world_rotation
        self.p1 = new.p1
        self.p2 = new.p2

    def draw(
        self, img: np.ndarray, settings: PnpDrawSettings = None
    ) -> np.ndarray:
        settings0 = settings if settings is not None else PnpDrawSettings()
        result = draw_pts2d(
            img=img,
            pts2d=self.kpt_2d,
            color=settings0.corners_line_color,
            radius=settings0.corners_pt_radius,
        )
        result = draw_corners(
            img=result,
            corner_2d=self.corner_2d,
            color=settings0.corners_line_color,
            thickness=settings0.corners_line_thickness,
        )
        if self.p1 is not None and self.p2 is not None:
            result = PnP_Model.draw_line(
                img=result,
                p1=self.p1,
                p2=self.p2,
                line_color=settings0.direction_line_color,
                point_color=settings0.direction_point_color,
                thickness=settings0.direction_line_thickness,
            )
        return result


class PnpPredictionList(
    BasicLoadableHandler["PnpPredictionList", "PnpPrediction"],
    BasicHandler["PnpPredictionList", "PnpPrediction"],
):
    def __init__(self, pred_list: List[PnpPrediction] = None):
        """List of PnpPrediction data. There are some useful methods implemented in
        this class for working with PnpPrediction data.

        Args:
            pred_list (List[PnpPrediction], optional):
                List of PnpPrediction objects.
                Empty list when None.
        """
        super().__init__(obj_type=PnpPrediction, obj_list=pred_list)
        self.pred_list = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> PnpPredictionList:
        return PnpPredictionList(
            [PnpPrediction.from_dict(item_dict) for item_dict in dict_list]
        )

    def draw(
        self, img: np.ndarray, settings: PnpDrawSettings = None
    ) -> np.ndarray:
        result = img.copy()
        for pred in self:
            result = pred.draw(img=result, settings=settings)
        return result


class PVNetFrameResult(BasicLoadableObject["PVNetFrameResult"]):
    def __init__(
        self,
        frame: str,
        pred_list: PnpPredictionList = None,
        test_name: str = None,
        model_name: str = None,
    ):
        """A collection of all pnp data and metadata in a single frame.

        Args:
            frame (str):
                The name of the frame. Usually an image filename.
                When working with images loaded from a video, the frame number
                can be used instead.
                When working with a live stream of images, such as a webcamera
                stream, a timestamp can be used here instead.
                This value is meant strictly for identification purposes.
            pred_list (PnpPredictionList, optional):
                PnpPredictionList object.
                Empty PnpPredictionList when None.
            test_name (str, optional):
                The name of the test data used to generate this prediction
                data.
                This can be the name of an image set the name of a video file,
                or some kind of description of what makes the test data unique.
                Defaults to None.
            model_name (str, optional):
                The name of the trained model that was used when inferring the
                test data to get this prediction data. This is usually a string
                that contains the name of the network, important
                hyperparameters used during training, and/or a timestamp
                indicating when the model was trained.
                Defaults to None.
        """
        super().__init__()
        self.frame = frame
        self.pred_list = pred_list \
            if pred_list is not None else PnpPredictionList()
        self.test_name = test_name
        self.model_name = model_name

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "pred_list": self.pred_list.to_dict_list(),
            "test_name": self.test_name,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> PVNetFrameResult:
        return PVNetFrameResult(
            frame=item_dict["frame"],
            pred_list=PnpPredictionList.from_dict_list(item_dict["pred_list"]),
            test_name=item_dict["test_name"],
            model_name=item_dict["model_name"],
        )

    def draw(
        self, img: np.ndarray, settings: PnpDrawSettings = None
    ) -> np.ndarray:
        """Draws all pvnet prediction data for a given frame onto an image.

        Args:
            img (np.ndarray): Input image.
            settings (PnpDrawSettings, optional):
                Pnp drawing settings.
                Defaults settings are used when None.

        Returns:
            np.ndarray: [description]
        """
        return self.pred_list.draw(img=img, settings=settings)


class PVNetFrameResultList(
    BasicLoadableHandler["PVNetFrameResultList", "PVNetFrameResult"],
    BasicHandler["PVNetFrameResultList", "PVNetFrameResult"],
):
    def __init__(self, result_list: List[PVNetFrameResult] = None):
        """All prediction and metadata for all frames.

        Args:
            result_list (List[PVNetFrameResult], optional):
                List of PVNetFrameResult objects.
                Empty list when None.
        """
        super().__init__(obj_type=PVNetFrameResult, obj_list=result_list)
        self.result_list = self.obj_list

    @property
    def model_names(self) -> List[str]:
        result = list(set([datum.model_name for datum in self]))
        result.sort()
        return result

    @property
    def test_names(self) -> List[str]:
        result = list(set([datum.test_name for datum in self]))
        result.sort()
        return result

    @property
    def frames(self) -> List[str]:
        result = list(set([datum.frame for datum in self]))
        result.sort()
        return result

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> PVNetFrameResultList:
        return PVNetFrameResultList(
            [PVNetFrameResult.from_dict(item_dict) for item_dict in dict_list]
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.to_dict_list())


class PVNetInferer:
    def __init__(self, weight_path: str, vote_dim: int = 18, seg_dim: int = 2):
        """Inferer for the PVNet model.
        This model can only detect one object at a time.
        In order to detect multiple objects, use the model together with a
        bbox model.

        Args:
            weight_path (str): Path to trained weights.
            vote_dim (int, optional):
                vote_dim parameter defined in network model.
                Only change this if you know what you are doing.
                Defaults to 18.
            seg_dim (int, optional):
                seg_dim parameter defined in network model.
                Only change this if you know what you are doing.
                Defaults to 2.
        """
        self.network = get_res_pvnet(vote_dim, seg_dim).cuda()
        custom_load_network(
            net=self.network, weight_path=weight_path, resume=True, strict=True
        )
        self.network.eval()
        self.transforms = make_transforms(is_train=False)

    def _predict(
        self, img: np.ndarray, bbox: BBox = None
    ) -> Dict[str, torch.cuda.Tensor]:
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
                    acceptable_types += f"\t{acceptable_type.__name__}"
                else:
                    acceptable_types += f"\n\t{acceptable_type.__name__}"
            raise TypeError(
                get_type_error_message(
                    func_name="PVNetInferer._predict",
                    acceptable_types=[np.ndarray, JpegImageFile],
                    unacceptable_type=type(img),
                    param_name="img",
                )
            )

        # Blackout Region Outside BBox If Given BBox
        if bbox is not None:
            img0 = np.array(img0)
            img0 = bbox.crop_and_paste(
                src_img=img0, dst_img=np.zeros_like(img0)
            )
            img0 = Image.fromarray(img0)

        # Infer
        img0 = self.transforms.on_image(img0)
        img0 = img0.reshape(tuple([1] + list(img0.shape)))
        img0 = torch.from_numpy(img0)
        img0 = img0.cuda()
        return self.network(img0)

    def predict(self, img: np.ndarray, bbox: BBox = None) -> PVNetPrediction:
        """Gets prediction data from an input image.

        Args:
            img (np.ndarray): Input image
            bbox (BBox, optional):
                Specify a bbox object if you would like to color all regions of
                the image outside of the bbox black.
                This approach can be used for detecting multiple objects in the
                same image given a list of bounding boxes.
                If this is not used, the prediction will be negatively affected
                when there are multiple target objects in the image.
                No modifications are made to the image when None.

        Returns:
            PVNetPrediction: Prediction data corresponding to pvnet results.
        """
        output = self._predict(img=img, bbox=bbox)
        return PVNetPrediction(
            seg=output["seg"].detach().cpu().numpy()[0],
            vertex=output["vertex"].detach().cpu().numpy()[0],
            mask=output["mask"].detach().cpu().numpy()[0],
            kpt_2d=output["kpt_2d"].detach().cpu().numpy()[0],
        )

    def infer_linemod_dataset(
        self,
        dataset: Linemod_Dataset,
        img_dir: str,
        blackout: bool = False,
        show_pbar: bool = True,
        show_preview: bool = False,
        video_save_path: str = None,
        dump_dir: str = None,
        accumulate_pred_dump: bool = True,
        pred_dump_path: str = None,
        test_name: str = None,
    ) -> PVNetFrameResultList:
        stream_writer = StreamWriter(
            show_preview=show_preview,
            video_save_path=video_save_path,
            dump_dir=dump_dir,
        )
        pbar = tqdm(total=len(dataset.images), unit="image(s)") \
            if show_pbar else None
        frame_result_list = (
            PVNetFrameResultList()
            if pred_dump_path is not None or accumulate_pred_dump
            else None
        )
        for linemod_image in dataset.images:
            file_name = get_filename(linemod_image.file_name)
            if pbar is not None:
                pbar.set_description(file_name)
            img_path = f"{img_dir}/{file_name}"
            img = Image.open(img_path)

            orig_img = np.asarray(img)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            result = orig_img.copy()

            gt_ann = dataset.annotations.get(image_id=linemod_image.id)[0]
            if blackout:
                xmin = int(
                    gt_ann.corner_2d.to_numpy(demarcation=True)[:, 0].min()
                )
                xmax = int(
                    gt_ann.corner_2d.to_numpy(demarcation=True)[:, 0].max()
                )
                ymin = int(
                    gt_ann.corner_2d.to_numpy(demarcation=True)[:, 1].min()
                )
                ymax = int(
                    gt_ann.corner_2d.to_numpy(demarcation=True)[:, 1].max()
                )
                bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                bbox = bbox.clip_at_bounds(frame_shape=result.shape)
                result = bbox.crop_and_paste(
                    src_img=result, dst_img=np.zeros_like(result)
                )
            else:
                bbox = None
            pred = self.predict(img=img, bbox=bbox)
            if frame_result_list is not None:
                frame_result = PVNetFrameResult(
                    frame=file_name,
                    pred_list=PVNetPredictionList([pred]),
                    test_name=test_name,
                )
                frame_result_list.append(frame_result)

            gt_kpt_3d = gt_ann.fps_3d.copy()
            gt_kpt_3d.append(gt_ann.center_3d)
            pose_pred = pred.to_pose_pred(gt_kpt_3d=gt_kpt_3d, K=gt_ann.K)
            corner_2d_gt = pvnet_pose_utils.project(
                gt_ann.corner_3d.to_numpy(),
                gt_ann.K.to_matrix(),
                np.array(gt_ann.pose.to_list()),
            )
            corner_2d_pred = pred.to_corner_2d_pred(
                gt_corner_3d=gt_ann.corner_3d, K=gt_ann.K, pose_pred=pose_pred
            )

            result = draw_pts2d(
                img=result, pts2d=gt_ann.fps_2d.to_numpy(),
                color=(0, 255, 0), radius=3
            )
            result = draw_corners(
                img=result, corner_2d=corner_2d_gt,
                color=(0, 255, 0), thickness=2
            )
            result = draw_pts2d(
                img=result, pts2d=pred.kpt_2d, color=(0, 0, 255), radius=2
            )
            result = draw_corners(
                img=result, corner_2d=corner_2d_pred,
                color=(0, 0, 255), thickness=2
            )
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
        kpt_3d: np.ndarray,
        corner_3d: np.ndarray,
        K: np.ndarray,
        camera_translation: np.ndarray = None,
        camera_quaternion: np.ndarray = None,
        distortion: np.ndarray = None,
        line_start_point3d: np.ndarray = None,
        line_end_point_3d: np.ndarray = None,
        units_per_meter: float = 1.0,
        blackout: bool = False,
        dsize: (int, int) = None,
        show_pbar: bool = True,
        leave_pbar: bool = False,
        preserve_dump_img_filename: bool = True,
        accumulate_pred_dump: bool = True,
        pred_dump_path: str = None,
        test_name: str = None,
        model_name: str = None,
        stream_writer: StreamWriter = None,
        leave_stream_writer_open: bool = False,
    ) -> PVNetFrameResultList:
        pbar = (
            tqdm(total=len(dataset.images), unit="image(s)", leave=leave_pbar)
            if show_pbar
            else None
        )
        frame_result_list = (
            PVNetFrameResultList()
            if pred_dump_path is not None or accumulate_pred_dump
            else None
        )
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

            anns = dataset.annotations.get_annotations_from_imgIds(
                [coco_image.id]
            )
            pred_list = PnpPredictionList()
            for ann in anns:
                working_img = orig_img.copy()
                if dsize is not None:
                    bbox = ann.bbox.resize(
                        orig_frame_shape=[orig_img_h, orig_img_w, 3],
                        new_frame_shape=[img_h, img_w, 3],
                    )
                else:
                    bbox = ann.bbox.copy()
                if blackout:
                    bbox = bbox.clip_at_bounds(frame_shape=working_img.shape)
                    working_img = bbox.crop_and_paste(
                        src_img=working_img, dst_img=np.zeros_like(working_img)
                    )
                pred = self.predict(
                    img=working_img, bbox=bbox if blackout else None
                )
                pred_list.append(
                    pred.to_pnp_pred(
                        gt_kpt_3d=kpt_3d,
                        corner_3d=corner_3d,
                        K=K,
                        camera_translation=camera_translation,
                        camera_quaternion=camera_quaternion,
                        distortion=distortion,
                        line_start_point3d=line_start_point3d,
                        line_end_point_3d=line_end_point_3d,
                        units_per_meter=units_per_meter,
                    )
                )
            if frame_result_list is not None:
                frame_result = PVNetFrameResult(
                    frame=file_name,
                    pred_list=pred_list,
                    test_name=test_name,
                    model_name=model_name,
                )
                frame_result_list.append(frame_result)
            result = pred_list.draw(img=orig_img)
            if stream_writer is not None:
                stream_writer.step(
                    img=result,
                    file_name=file_name
                    if preserve_dump_img_filename else None,
                )
            if pbar is not None:
                pbar.update()
        if stream_writer is not None and not leave_stream_writer_open:
            stream_writer.close()
        if pred_dump_path is not None:
            frame_result_list.save_to_path(pred_dump_path, overwrite=True)
        if pbar is not None:
            pbar.close()
        return frame_result_list


def pvnet_infer(
    kpt_3d: np.ndarray,
    corner_3d: np.ndarray,
    K: np.ndarray,  # Required
    weight_path: str,
    model_name: str,
    dataset: COCO_Dataset,
    test_name: str,  # Required for wrapper
    vote_dim: int = 18,
    seg_dim: int = 2,  # Optional
    camera_translation: np.ndarray = None,
    camera_quaternion: np.ndarray = None,
    distortion: np.ndarray = None,
    line_start_point3d: np.ndarray = None,
    line_end_point_3d: np.ndarray = None,
    units_per_meter: float = 1.0,
    blackout: bool = False,
    dsize: (int, int) = None,
    show_pbar: bool = True,
    accumulate_pred_dump: bool = True,
    stream_writer: StreamWriter = None,
    leave_stream_writer_open: bool = False,
) -> PVNetFrameResultList:
    inferer = PVNetInferer(
        weight_path=weight_path, vote_dim=vote_dim, seg_dim=seg_dim
    )
    pred_data = inferer.infer_coco_dataset(
        dataset=dataset,
        kpt_3d=kpt_3d,
        corner_3d=corner_3d,
        K=K,
        accumulate_pred_dump=accumulate_pred_dump,
        pred_dump_path=None,
        preserve_dump_img_filename=True,
        camera_translation=camera_translation,
        camera_quaternion=camera_quaternion,
        distortion=distortion,
        line_start_point3d=line_start_point3d,
        line_end_point_3d=line_end_point_3d,
        units_per_meter=units_per_meter,
        blackout=blackout,
        dsize=dsize,
        show_pbar=show_pbar,
        leave_pbar=False,
        stream_writer=stream_writer,
        leave_stream_writer_open=leave_stream_writer_open,
        model_name=model_name,
        test_name=test_name,
    )
    del inferer
    return pred_data


def infer_tests_pvnet(
    weight_path: str,
    model_name: str,
    dataset: COCO_Dataset,
    test_name: str,
    kpt_3d: np.ndarray,
    corner_3d: np.ndarray,
    K: np.ndarray,
    show_pbar: bool = True,
    vote_dim: int = 18,
    seg_dim: int = 2,
    camera_translation: np.ndarray = None,
    camera_quaternion: np.ndarray = None,
    distortion: np.ndarray = None,
    line_start_point3d: np.ndarray = None,
    line_end_point_3d: np.ndarray = None,
    units_per_meter: float = 1.0,
    blackout: bool = False,
    dsize: (int, int) = None,
    show_preview: bool = False,
    data_dump_dir: str = None,
    video_dump_dir: str = None,
    img_dump_dir: str = None,
    skip_if_data_dump_exists: bool = False,
):
    infer_tests_wrapper(
        weight_path=weight_path,
        model_name=model_name,
        dataset=dataset,
        test_name=test_name,
        handler_constructor=PVNetFrameResultList,
        data_dump_dir=data_dump_dir,
        video_dump_dir=video_dump_dir,
        img_dump_dir=img_dump_dir,
        skip_if_data_dump_exists=skip_if_data_dump_exists,
        show_preview=show_preview,
        show_pbar=show_pbar,
    )(pvnet_infer)(
        kpt_3d=kpt_3d,
        corner_3d=corner_3d,
        K=K,
        vote_dim=vote_dim,
        camera_translation=camera_translation,
        camera_quaternion=camera_quaternion,
        distortion=distortion,
        line_start_point3d=line_start_point3d,
        line_end_point_3d=line_end_point_3d,
        units_per_meter=units_per_meter,
        blackout=blackout,
        dsize=dsize,
        show_pbar=show_pbar,
    )
