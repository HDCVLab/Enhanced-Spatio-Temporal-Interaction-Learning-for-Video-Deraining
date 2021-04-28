import argparse


def _get_test_opt():
    parser = argparse.ArgumentParser(description='Evaluate performance on test set')

    # network
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
    parser.add_argument('--use_bilstm', action='store_true')
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--input_residue', action='store_true')
    parser.add_argument('--compress_channels', type=int, default=8, help='number of channels in R_CLSTM.')
    parser.add_argument('--F_npic', action='store_true', default=False, help='if True, output n images from fine net; '
                                                                             'otherwise output the one in the middle.')
    parser.add_argument('--torch_home', type=str, default=None)

    # validation
    parser.add_argument('--val_mode', type=str, default='all', help='validation on the frame in the middle (mid) or '
                                                                    'all the frames (all) of coarse network.')
    # paths
    parser.add_argument('--eval_file', type=str, required=True, help='the path of indexfile',
                        default='/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json')
    parser.add_argument('--checkpoint_dir_C', required=False, help="the directory to save the checkpoints of coarse net",
                        default='./checkpoint/C')
    parser.add_argument('--checkpoint_dir_F', required=False, help="the directory to save the checkpoints of finenet",
                        default='./checkpoint/F')
    parser.add_argument('--data_root', type=str, required=True, help="the root path of image data.",
                        default='/media/hdd/derain/NTU-derain')
    parser.add_argument('--out_dir', type=str, required=True, help="output dir root.")

    # misc
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--loadckpt', action='store_true')
    parser.add_argument('--loadckpt_C', type=str, help='pretrain weights of coarse net.')
    parser.add_argument('--loadckpt_F', type=str, help='pretrain weights of fine net.')
    parser.add_argument('--use_cuda', type=bool, default=True)

    return parser.parse_args()

