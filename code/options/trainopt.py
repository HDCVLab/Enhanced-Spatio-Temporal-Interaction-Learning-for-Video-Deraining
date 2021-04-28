
import argparse


def _get_train_opt():
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation')
    # network
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--refinenet', type=str, default='R_CLSTM_5')
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--use_bilstm', action='store_true')
    parser.add_argument('--input_residue', action='store_true')
    parser.add_argument('--compress_channels', type=int, default=8, help='number of channels in R_CLSTM.')
    parser.add_argument('--crop_size', default=224, type=int, help='patch size to crop for training.')

    parser.add_argument('--freeze_net_C', action='store_true', default=False, help='freeze coarse net or not.')
    parser.add_argument('--freeze_net_F', action='store_true', default=False, help='freeze fine net or not.')
    parser.add_argument('--F_npic', action='store_true', default=False, help='if True, output n images from fine net; '
                                                                             'otherwise output the one in the middle.')
    parser.add_argument('--torch_home', type=str, default=None)

    # optimization
    parser.add_argument('--resume', action='store_true', default=False, help='continue training the model')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter used in the Optimizer.')
    parser.add_argument('--epsilon', default=0.0001, type=float, help='epsilon')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    #     coarse network specifics
    parser.add_argument('--lr_C', default=0.0001, type=float, help='initial learning rate of coarse network.')
    #     fine network specifics
    parser.add_argument('--lr_F', default=0.0001, type=float, help='initial learning rate of fine network.')

    # validation
    parser.add_argument('--eval_num_batch', default=50, type=int, help='number of epochs')
    parser.add_argument('--eval_on_the_fly', type=bool, default=True)
    parser.add_argument('--eval_interval', default=2, type=int, help='number of epochs')
    parser.add_argument('--val_mode', type=str, default='all', help='validation on the frame in the middle (mid) or '
                                                                    'all the frames (all) of coarse network.')
    # logging
    parser.add_argument('--log_interval', type=int, default=10, help='logging per number of iterations.')
    parser.add_argument('--logdir', required=True, help="the directory to save logs and checkpoints")

    # paths
    parser.add_argument('--train_file', type=str, required=True, help='the path of indexfile',
                        default='/home/dxli/workspace/derain/proj/data/Dataset_Training_Synthetic.json')
    parser.add_argument('--eval_file', type=str, required=True, help='the path of indexfile',
                        default='/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json')
    parser.add_argument('--checkpoint_dir_C', required=True, help="the directory to save the checkpoints of coarse net",
                        default='./checkpoint/C')
    parser.add_argument('--checkpoint_dir_F', required=True, help="the directory to save the checkpoints of finenet",
                        default='./checkpoint/F')
    parser.add_argument('--data_root', type=str, required=True, help="the root path of image data.",
                        default='/media/hdd/derain/NTU-derain')

    # misc
    parser.add_argument('--loadckpt', type=str)
    parser.add_argument('--use_cuda', type=bool, default=True)

    return parser.parse_args()

