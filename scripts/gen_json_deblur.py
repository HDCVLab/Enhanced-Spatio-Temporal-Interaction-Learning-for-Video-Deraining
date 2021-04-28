import os
import json

# num_instances = 8
# out_dir = '/home/dxli/workspace/derain/proj/data'
out_dir = '/home/dxli/workspace/videodeblur/data'

# root = '/media/hdd/derain/NTU-derain/Dataset_Training_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_Synthetic'
# root = '/media/hdd/derain/NTU-derain/Dataset_Testing_RealRain'
root = '/media/hdd/videodeblur/videodeblurring/quantitative_datasets/test'

out_filepath = os.path.join(out_dir, os.path.split(root)[-1] + '.json')

all_dirs = sorted([d for d in [os.path.join(root, x) for x in os.listdir(root)]])

entry_list = list()

for d in all_dirs:
    entry = []

    prefix, filename = os.path.split(d)

    gt_dirpath = os.path.join(d, 'GT')
    blur_dirpath = os.path.join(d, 'input')

    frame_names = sorted([f for f in os.listdir(gt_dirpath) if f.endswith('jpg')])

    for fn in frame_names:
        blur_filepath = os.path.join(blur_dirpath, fn)
        gt_filepath = os.path.join(gt_dirpath, fn)

        entry.append({'blur': blur_filepath, 'gt': gt_filepath})

    entry_list.append(entry)

json.dump(entry_list, open(out_filepath, 'w'))



