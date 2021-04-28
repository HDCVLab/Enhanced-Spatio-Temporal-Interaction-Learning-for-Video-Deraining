#!/usr/bin/env bash

cd ..
python test.py --data_root /media/hdd/derain/RainSyn25/ \
               --eval_file /home/dxli/workspace/derain/proj/code/data/frames_light_test_PNG.json \
               --backbone resnet18 \
               --refinenet R_CLSTM_5 \
               --use_bilstm \
               --input_residue \
               --val_mode all \
	       --F_npic \
               --loadckpt \
               --loadckpt_C /home/dxli/workspace/derain/proj/code/best_checkpoints/light/C/checkpoints_best_1586.pth.tar\
               --loadckpt_F /home/dxli/workspace/derain/proj/code/best_checkpoints/light/F/checkpoints_best_1586.pth.tar\
               --out_dir /home/dxli/workspace/derain/proj/out/light/1586



#def _get_test_opt():
#    parser = argparse.ArgumentParser(description='Evaluate performance on test set')
#
#    # network
#    parser.add_argument('--backbone', type=str, default='resnet18')
#    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
#    parser.add_argument('--use_bilstm', action='store_true')
#    parser.add_argument('--use_bn', action='store_true', default=False)
#    parser.add_argument('--input_residue', action='store_true')
#    parser.add_argument('--compress_channels', type=int, default=8, help='number of channels in R_CLSTM.')
#
#    # validation
#    parser.add_argument('--val_mode', type=str, default='mid', help='validation on the frame in the middle (mid) or '
#                                                                    'all the frames (all) of coarse network.')
#    # paths
#    parser.add_argument('--eval_file', type=str, required=True, help='the path of indexfile',
#                        default='/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json')
#    parser.add_argument('--checkpoint_dir_C', required=True, help="the directory to save the checkpoints of coarse net",
#                        default='./checkpoint/C')
#    parser.add_argument('--checkpoint_dir_F', required=True, help="the directory to save the checkpoints of finenet",
#                        default='./checkpoint/F')
#    parser.add_argument('--data_root', type=str, required=True, help="the root path of image data.",
#                        default='/media/hdd/derain/NTU-derain')
#
#    # misc
#    parser.add_argument('--loadckpt', action='store true')
#    parser.add_argument('--loadckpt_C', type=str, help='pretrain weights of coarse net.')
#    parser.add_argument('--loadckpt_F', type=str, help='pretrain weights of fine net.')
#    parser.add_argument('--use_cuda', type=bool, default=True)
#
#    return parser.parse_args()
