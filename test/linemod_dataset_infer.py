from clean_pvnet.infer.pvnet_inferer import PVNetInferer
from annotation_utils.linemod.objects import Linemod_Dataset

inferer = PVNetInferer(
    # weight_path='/home/clayton/workspace/git/clean-pvnet/data/model/pvnet/custom/99.pth',
    # weight_path='/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_weights/194.pth',
    weight_path='/home/clayton/workspace/prj/data_keep/data/weights/pvnet_hsr/20201119/59.pth'
)
# img_dir = '/home/clayton/workspace/git/pvnet-rendering/test/renders1'
img_dir = '/home/clayton/workspace/prj/data_keep/data/misc_dataset/darwin_datasets/nihonbashi2/organized'
linemod_dataset = Linemod_Dataset.load_from_path(f'{img_dir}/train.json')
linemod_dataset.images = linemod_dataset.images[:100]
inferer.infer_linemod_dataset(
    dataset=linemod_dataset, img_dir=img_dir, blackout=True,
    video_save_path='train_infer.avi', show_preview=True
)