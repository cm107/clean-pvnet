from tqdm import tqdm
from common_utils.path_utils import get_all_files_of_extension, get_filename
from common_utils.file_utils import make_dir_if_not_exists
from clean_pvnet.infer.pvnet_inferer import PVNetFrameResultList
from pasonatron.pvnet.infer import PVNetPredictionPairList, TwoStepPVNetFrameResultList

fixed_data_combined_root = '/home/clayton/workspace/prj/data_keep/data/toyota/from_toyota/20201017/20201221_robot_2F_north/2F_North_NS_1-data_combined'
fixed_pvnet_infer_data = PVNetFrameResultList.load_from_path(f'{fixed_data_combined_root}/pvnet_infer.json')
fixed_2step_infer_pairs_data = PVNetPredictionPairList.load_from_path(f'{fixed_data_combined_root}/2step_pairs_infer.json')
fixed_2step_infer_data = TwoStepPVNetFrameResultList.load_from_path(f'{fixed_data_combined_root}/2step_infer.json')

infer_data_dump_dir = 'infer_data_dump'
fixed_infer_dump_dir = 'infer_data_dump-fixed'
make_dir_if_not_exists(fixed_infer_dump_dir)

dump_paths = get_all_files_of_extension(infer_data_dump_dir, 'json')
pbar = tqdm(total=len(dump_paths), unit='dump(s)', leave=True)
for dump_path in dump_paths:
    dump_data = PVNetFrameResultList.load_from_path(dump_path)
    fixed_dump_data = PVNetFrameResultList()
    for model_name in dump_data.model_names:
        fixed_model_data = fixed_pvnet_infer_data.get(model_name=model_name)
        model_data = dump_data.get(model_name=model_name)
        for test_name in model_data.test_names:
            test_data = model_data.get(test_name=test_name)
            if test_name in fixed_model_data.test_names:
                fixed_test_data = fixed_model_data.get(test_name=test_name)
                for frame in fixed_test_data.frames:
                    # frame_data = test_data.get(frame=frame)
                    # for datum in frame_data:
                    #     datum.test_name = 'old_' + datum.test_name
                    #     fixed_dump_data.append(datum)
                    fixed_frame_data = fixed_test_data.get(frame=frame)
                    for datum in fixed_frame_data:
                        fixed_dump_data.append(datum)
            else:
                for frame in test_data.frames:
                    frame_data = test_data.get(frame=frame)
                    for datum in frame_data:
                        fixed_dump_data.append(datum)
    fixed_dump_data.save_to_path(f'{fixed_infer_dump_dir}/{get_filename(dump_path)}', overwrite=True)
    pbar.update()
pbar.close()