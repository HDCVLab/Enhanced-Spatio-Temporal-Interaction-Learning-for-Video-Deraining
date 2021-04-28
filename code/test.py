import sys
import time

import torch.nn as nn
import torch.nn.parallel
from torchvision import transforms

import utils
from dataset import ntu_dataset
from models import modules
from models import net
from models.backbone_dict import backbone_dict
from options import get_args
from utils import *

import logging

# cudnn.benchmark = True
logging_root = './test_log'
if not os.path.exists(logging_root):
    os.mkdir(logging_root)

logging.basicConfig(filename=os.path.join(logging_root, '{}.log'.format(int(time.time()))), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# cudnn optimization
torch.backends.cudnn.benchmark = True


def validate(args, net_C, net_F, out_dir):
    gt_dir = os.path.join(out_dir, 'gt')
    cout_dir = os.path.join(out_dir, 'C_out')
    fout_dir = os.path.join(out_dir, 'F_out')

    makedir(gt_dir)
    makedir(cout_dir)
    makedir(fout_dir)

    # manually release GPU memory.
    torch.cuda.empty_cache()

    test_loader = get_test_loader(args)

    net_C.eval()
    net_F.eval()

    all_out_psnr_C = AverageMeter()
    all_inp_psnr_C = AverageMeter()
    all_out_psnr_F = AverageMeter()
    all_inp_psnr_F = AverageMeter()
    time_C = AverageMeter()
    time_F = AverageMeter()
    time_batch = AverageMeter()

    end = time.time()

    for batch_idx, sample in enumerate(test_loader):
        inp_videos, gt_videos = sample[0].cuda(), sample[1].cuda()  # (b,c,d,w,h)
        batch_size, _, nf, _, _ = inp_videos.shape

        # model forward
        with torch.no_grad():
            model_out_C = net_C(inp_videos)  # (b, c, d, h, w)
            c_now = time.time()
            time_C.update(c_now - end)

            model_out_F = net_F(model_out_C, inp_videos)  # (b, c, 1, h, w)
            time_F.update(time.time() - c_now)

            # batch time
            time_batch.update(time.time() - end)

        # validation range for output of coarse network
        if args.val_mode == 'all':
            val_range = range(nf)
        elif args.val_mode == 'mid':
            val_range = range(nf // 2, nf // 2 + 1)
        else:
            raise ValueError("invalid validation mode, must be 'all' or 'mid', got {}".format(args.val_mode))

        model_out_C = clamp_on_imagenet_stats(model_out_C)
        # model_out_F = clamp_on_imagenet_stats(model_out_F.unsqueeze(2)).squeeze(2)
        model_out_F = clamp_on_imagenet_stats(model_out_F)

        for i in range(batch_size):
            # validate for output of coarse network
            for j in val_range:
                out_C, gt_C, inp_C = model_out_C[i, :, j, :, :], gt_videos[i, :, j, :, :], inp_videos[i, :, j, :, :]

                out_psnr_C, inp_psnr_C = calculate_psnr(gt_C, out_C), calculate_psnr(gt_C, inp_C)

                all_out_psnr_C.update(out_psnr_C)
                all_inp_psnr_C.update(inp_psnr_C)

                if args.F_npic:
                    # validate for output of fine network
                    out_F, gt_F, inp_F = model_out_F[i, :, j, :, :], gt_videos[i, :, j, :, :], model_out_C[i, :, j, :,
                                                                                               :]
                    out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

                    all_out_psnr_F.update(out_psnr_F)
                    all_inp_psnr_F.update(inp_psnr_F)

                    save_image(gt_C, os.path.join(gt_dir, '{}_{}_{}.png'.format(batch_idx, i, j)))
                    save_image(out_C, os.path.join(cout_dir, '{}_{}_{}.png'.format(batch_idx, i, j)))
                    save_image(out_F, os.path.join(fout_dir, '{}_{}_{}.png'.format(batch_idx, i, j)))

            if not args.F_npic:
                raise ValueError('F_npic = False.')
                # validate for output of fine network
                out_F, gt_F, inp_F = model_out_F[i, :, :, :], gt_videos[i, :, nf // 2, :, :], model_out_C[i, :, nf // 2,
                                                                                              :, :]
                out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

                all_out_psnr_F.update(out_psnr_F)
                all_inp_psnr_F.update(inp_psnr_F)

        if batch_idx % 1 == 0:
            logging.info('[{batch}/{total_batch}]\tbth. out_C PSNR: {boc.val:.3f},\t inp_C PSNR: {binpc.val:.3f}\t'
                         'out_F PSNR: {bof.val:.4f},\tC time: {tc.val:.3f}\tF time: {tf.val:.3f}\t batch time: {tb.val:.3f}\n'
                         '[{batch}/{total_batch}]\tavg. out_C PSNR: {boc.avg:.3f},\t inp_C PSNR: {binpc.avg:.3f}\t'
                         'out_F PSNR: {bof.avg:.3f},\tC time: {tc.avg:.3f}\tF time: {tf.avg:.3f}\t batch time: {tb.avg:.3f}'.format(batch=batch_idx,
                                                                                                                      boc=all_out_psnr_C,
                                                                                                                      binpc=all_inp_psnr_C,
                                                                                                                      bof=all_out_psnr_F,
                                                                                                                      binpf=all_inp_psnr_F,
                                                                                                                      tc=time_C,
                                                                                                                      tf=time_F,
                                                                                                                      tb=time_batch,
                                                                                                                                    total_batch=len(test_loader)
                                                                                                                      ))

        torch.cuda.empty_cache()

        end = time.time()

    return all_out_psnr_C.avg, all_out_psnr_F.avg


def test(args):
    net_C, net_F = build_model(args)

    net_C = nn.DataParallel(net_C).cuda()
    net_F = nn.DataParallel(net_F).cuda()

    load_checkpoint(args, net_C, pretrain_ckpt=args.loadckpt_C)
    load_checkpoint(args, net_F, pretrain_ckpt=args.loadckpt_F)

    # out_dir = '/home/dxli/workspace/derain/proj/out/ntu'
    out_dir = args.out_dir

    validate(args, net_C=net_C, net_F=net_F, out_dir=out_dir)


def main():
    args = get_args('test')
    test(args)


if __name__ == '__main__':
    main()
