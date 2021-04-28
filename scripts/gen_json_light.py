import os
import json

num_instances = 8
out_dir = '/home/dxli/workspace/derain/proj/data'

# root = '/media/hdd/derain/RainSyn25/frames_heavy_test_JPEG'
# root = '/media/hdd/derain/RainSyn25/frames_light_train_JPEG'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'

# root = '/home/dxli/Desktop/video_rain_light/train'
# out_filepath = os.path.join(out_dir, 'frames_light_train_JPEG' + '.json')

root = '/home/dxli/Desktop/video_rain_light/test'
out_filepath = os.path.join(out_dir, 'frames_light_test' + '.json')

gt_prefix = 'gt'
input_prefix = 'input'

gt_dir = os.path.join(root, gt_prefix)
input_dir = os.path.join(root, input_prefix)

all_dirs = sorted([os.path.join(gt_dir, x) for x in os.listdir(gt_dir)])

entry_list = list()

for gt_d in all_dirs:
    ip_d = os.path.join(input_dir, os.path.split(gt_d)[-1])

    entry = []
    c_entry = []

    gt_frame_names = sorted(os.listdir(gt_d), key=lambda x: int(x[:-4]))

    for gt_fn in gt_frame_names:

        # rf_filepath = os.path.join(d, rf_fn)
        gtc_filepath = os.path.join(gt_d, gt_fn)
        rfc_filepath = os.path.join(ip_d, gt_fn)

        # entry.append({'rf': rf_filepath, 'gt': gt_filepath})
        c_entry.append({'rain': rfc_filepath, 'gt': gtc_filepath})

    # entry_list.append(entry)
    entry_list.append(c_entry)

json.dump(entry_list, open(out_filepath, 'w'))



