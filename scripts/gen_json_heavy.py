import os
import json

num_instances = 8
out_dir = '/home/dxli/workspace/derain/proj/data'

# root = '/media/hdd/derain/RainSyn25/frames_heavy_test_JPEG'
root = '/media/hdd/derain/RainSyn25/frames_light_train_JPEG'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'

out_filepath = os.path.join(out_dir, os.path.split(root)[-1] + '.json')


all_dirs = sorted([os.path.join(root, x) for x in os.listdir(root)])

entry_list = list()

for d in all_dirs:
    entry = []
    c_entry = []

    all_files = os.listdir(d)

    gt_frame_names = sorted([f for f in os.listdir(d) if f.startswith('gt-')], key=lambda x: int(x[3:-4]))

    for gt_fn in gt_frame_names:
        pic_id = gt_fn[3:-4]

        # gt_filepath = os.path.join(d, gt_fn)

        # rf_fn = 'rf-' + pic_id + '.jpg'
        gtc_fn = 'gtc-' + pic_id + '.jpg'
        rfc_fn = 'rfc-' + pic_id + '.jpg'

        # rf_filepath = os.path.join(d, rf_fn)
        gtc_filepath = os.path.join(d, gtc_fn)
        rfc_filepath = os.path.join(d, rfc_fn)

        # entry.append({'rf': rf_filepath, 'gt': gt_filepath})
        c_entry.append({'rf': rfc_filepath, 'gt': gtc_filepath})

    # entry_list.append(entry)
    entry_list.append(c_entry)

json.dump(entry_list, open(out_filepath, 'w'))



