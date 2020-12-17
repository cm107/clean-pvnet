import cv2
import numpy as np
from tqdm import tqdm
from clean_pvnet.infer.pvnet_inferer import PVNetInferer, PnpPredictionList, \
    PVNetFrameResult, PVNetFrameResultList, PnpDrawSettings
from annotation_utils.linemod.objects import Linemod_Dataset
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.path_utils import get_valid_image_paths
from streamer.recorder.stream_writer import StreamWriter

# Sample hyperparameters from Linemod dataset
inferer = PVNetInferer(weight_path='/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/499.pth')
linemod_dataset = Linemod_Dataset.load_from_path('/home/clayton/workspace/prj/data_keep/data/misc_dataset/clayton_datasets/combined/output.json')
kpt_3d, corner_3d, K = linemod_dataset.sample_3d_hyperparams(idx=0) # Should be constant
dsize = linemod_dataset.sample_dsize(idx=0) # Assuming constant resolution

# Define test dataset that you want to run inference on
img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/darwin20201209/coco_data'
ann_path = f'{img_dir}/coco_annotations.json'
test_dataset = COCO_Dataset.load_from_path(ann_path, img_dir=img_dir, strict=False)
test_dataset, _ = test_dataset.split_into_parts(ratio=[100, len(test_dataset.images)-100], shuffle=True)

# Define a StreamWriter for creating a preview window and saving video
stream_writer = StreamWriter(show_preview=True, video_save_path='example_pnp_inference.avi')
draw_settings = PnpDrawSettings()
draw_settings.direction_line_color = (255, 255, 0)

# Initialize Inference Dump Data Handler
pred_list = PVNetFrameResultList()

# Inference Loop
pbar = tqdm(total=len(test_dataset.images), unit='image(s)', leave=True)
for coco_image in test_dataset.images:
    pbar.set_description(coco_image.file_name)
    img = cv2.imread(coco_image.coco_url)

    anns = test_dataset.annotations.get(image_id=coco_image.id)
    pnp_pred_list = PnpPredictionList()
    for ann in anns:
        pvnet_pred = inferer.predict(img=img, bbox=ann.bbox)
        pnp_pred = pvnet_pred.to_pnp_pred(
            gt_kpt_3d=kpt_3d,
            corner_3d=corner_3d,
            K=K
        )
        pnp_pred_list.append(pnp_pred)
    frame_pred = PVNetFrameResult(
        frame=coco_image.file_name, test_name='example_dataset',
        model_name='example_model', pred_list=pnp_pred_list
    )
    result = frame_pred.draw(img=img, settings=draw_settings)
    stream_writer.step(img=result)
    pred_list.append(frame_pred)
    pbar.update()
pbar.close()

# Close StreamWriter
stream_writer.close()

# Save Inference Dump Data
pred_list.save_to_path('example_pnp_dump.json')