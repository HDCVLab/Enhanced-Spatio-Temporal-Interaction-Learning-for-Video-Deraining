import os
import json

num_instances = 8
out_dir = '/home/dxli/workspace/derain/proj/data'

root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'

out_filepath = os.path.join(out_dir, os.path.split(root)[-1] + '.json')


all_dirs = sorted([d for d in [os.path.join(root, x) for x in os.listdir(root)] if os.path.isdir(d)])


entry_list = list()

for d in all_dirs:
    entry = []

    prefix, filename = os.path.split(d)
    prefix = os.path.split(prefix)[-1]

    gt_dirname = filename
    # gt_dirname = filename[:-4] + 'GT'
    gt_dirpath = os.path.join(prefix, gt_dirname)

    frame_names = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))

    for fn in frame_names:
        rain_filepath = os.path.join(prefix, filename, fn)
        gt_filepath = os.path.join(gt_dirpath, fn)

        entry.append({'rain': rain_filepath, 'gt': rain_filepath})

    entry_list.append(entry)

json.dump(entry_list, open(out_filepath, 'w'))



