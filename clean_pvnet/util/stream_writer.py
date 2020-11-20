import cv2
import numpy as np
from typing import cast
from streamer.recorder import Recorder
from common_utils.file_utils import delete_all_files_in_dir, make_dir_if_not_exists
from common_utils.path_utils import get_next_dump_path, get_valid_image_paths, get_extension_from_path
from common_utils.cv_drawing_utils import SimpleVideoViewer

class VideoWriter:
    def __init__(self, save_path: str, fps: int=5):
        self._save_path = save_path
        self._fps = fps
        self.recorder = cast(Recorder, None)

    def write(self, img: np.ndarray):
        if self.recorder is None:
            img_h, img_w = img.shape[:2]
            self.recorder = Recorder(output_path=self._save_path, output_dims=(img_w, img_h), fps=self._fps)
        self.recorder.write(img)
    
    def close(self):
        self.recorder.close()

class ImageDumpWriter:
    def __init__(self, save_dir: str, clear: bool=True):
        self._save_dir = save_dir
        self._clear = clear
        self._first = True
        self._existing_extensions = []
    
    def init_save_dir(self):
        make_dir_if_not_exists(self._save_dir)
        if self._clear:
            delete_all_files_in_dir(self._save_dir, ask_permission=False)
        else:
            self._existing_extensions = list(set([get_extension_from_path(path) for path in get_valid_image_paths(self._save_dir)]))
        self._first = False

    def write(self, img: np.ndarray, file_name: str=None):
        if self._first:
            self.init_save_dir()
        if file_name is None:
            if len(self._existing_extensions) == 0:
                self._existing_extensions.append('png')
            dump_path = get_next_dump_path(
                dump_dir=self._save_dir, file_extension=self._existing_extensions[0],
                label_length=6, starting_number=0, increment=1
            )
        else:
            dump_path = f'{self._save_dir}/{file_name}'
        cv2.imwrite(dump_path, img)

class StreamWriter:
    def __init__(self, show_preview: bool=False, video_save_path: str=None, dump_dir: str=None):
        self.viewer = SimpleVideoViewer(preview_width=1000) if show_preview else None
        # self.video_save_path = video_save_path
        self.video_writer = VideoWriter(save_path=video_save_path, fps=5) if video_save_path is not None else None
        self.dump_writer = ImageDumpWriter(save_dir=dump_dir, clear=True) if dump_dir is not None else None
    
    def step(self, img: np.ndarray, file_name: str=None):
        if self.video_writer is not None:
            self.video_writer.write(img)
        if self.dump_writer is not None:
            self.dump_writer.write(img=img, file_name=file_name)
        if self.viewer is not None:
            self.viewer.show(img=img)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()