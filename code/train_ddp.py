import shutil
import sys
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from options import get_args
from utils import *


import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

logging_root = './train_log'
if not os.path.exists(logging_root):
    os.mkdir(logging_root)

logging.basicConfig(filename=os.path.join(logging_root, '{}.log'.format(int(time.time()))), filemode='w',
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# cudnn optimization
torch.backends.cudnn.benchmark = True


def validate(args, net_C, net_F):
    # manually release GPU memory.
    torch.cuda.empty_cache()

    test_loader = get_test_loader(args)

    net_C.eval()
    net_F.eval()

    batch_time = AverageMeter()
    end = time.time()

    all_out_psnr_C = AverageMeter()
    all_inp_psnr_C = AverageMeter()
    all_out_psnr_F = AverageMeter()
    all_inp_psnr_F = AverageMeter()
    time_C = AverageMeter()
    time_F = AverageMeter()
    time_batch = AverageMeter()

    for batch_idx, sample in enumerate(test_loader):
        if batch_idx > args.eval_num_batch:
            # only test first a couple of batches to save validation time during training.
            break

        inp_videos, gt_videos = sample[0], sample[1]  # (b,c,d,w,h)
        batch_size, _, nf, _, _ = inp_videos.shape

        inp_videos = inp_videos.cuda()
        gt_videos = gt_videos.cuda()

        # model forward
        with torch.no_grad():
            model_out_C = net_C(inp_videos)  # (b, c, d, h, w)
            model_out_C = clamp_on_imagenet_stats(model_out_C)
            c_now = time.time()
            time_C.update(c_now - end)

            model_out_F = net_F(model_out_C, inp_videos)  # (b, c, 1, h, w)
            model_out_F = clamp_on_imagenet_stats(model_out_F.unsqueeze(2)).squeeze(2)
            time_F.update(time.time() - c_now)

            batch_time.update(time.time() - end)
            end = time.time()

        # validation range for output of coarse network
        if args.val_mode == 'full':
            val_range = range(nf)
        elif args.val_mode == 'mid':
            val_range = range(nf // 2, nf // 2 + 1)
        else:
            raise ValueError("invalid validation mode, must be 'all' or 'mid', got {}".format(args.val_mode))

        for i in range(batch_size):
            # validate for output of coarse network
            for j in val_range:
                out_C, gt_C, inp_C = model_out_C[i, :, j, :, :], gt_videos[i, :, j, :, :], inp_videos[i, :, j, :, :]

                out_psnr_C, inp_psnr_C = calculate_psnr(gt_C, out_C), calculate_psnr(gt_C, inp_C)

                all_out_psnr_C.update(out_psnr_C)
                all_inp_psnr_C.update(inp_psnr_C)

            # validate for output of fine network
            out_F, gt_F, inp_F = model_out_F[i, :, :, :], gt_videos[i, :, nf // 2, :, :], model_out_C[i, :, nf // 2, :, :]

            out_psnr_F, inp_psnr_F = calculate_psnr(gt_F, out_F), calculate_psnr(gt_F, inp_F)

            all_out_psnr_F.update(out_psnr_F)
            all_inp_psnr_F.update(inp_psnr_F)

        if batch_idx % 2 == 0:
            logging.info('{batch}\tbth. out_C PSNR: {boc.val:.3f},\t inp_C PSNR: {binpc.val:.3f}\t'
                         'out_F PSNR: {bof.val:.4f},\t inp_F PSNR: {binpf.val:.3f}\n'
                         '{batch}\tavg. out_C PSNR: {boc.avg:.3f},\t inp_C PSNR: {binpc.avg:.3f}\t'
                         'out_F PSNR: {bof.avg:.3f},\t inp_F PSNR: {binpf.avg:.3f}'.format(batch=batch_idx,
                                                                            boc=all_out_psnr_C, binpc=all_inp_psnr_C,
                                                                            bof=all_out_psnr_F, binpf=all_inp_psnr_F,
                                                                            tc=time_C, tf=time_F, tb=time_batch))

        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.empty_cache()

    return all_out_psnr_C.avg, all_out_psnr_F.avg


def train(args, rank):
    print(f"Running basic DDP example on rank {rank}.")

    train_loader = get_train_loader(args)
    net_C, net_F = build_model(args)

    # net_C = nn.DataParallel(net_C)
    # net_F = nn.DataParallel(net_F)

    net_C = DDP(net_C.to(rank), device_ids=[rank])
    net_F = DDP(net_F.to(rank), device_ids=[rank])

    # start_epoch = max(load_checkpoint(args, net_C, args.checkpoint_dir_C),
    #                   load_checkpoint(args, net_F, args.checkpoint_dir_F)
    #                   )
    start_epoch = 0

    logging.info(net_C)
    logging.info(net_F)
    logging.info(args)

    # net_C.cuda()
    # net_F.cuda()

    optimizer_C = build_optimizer(model=net_C,
                                  learning_rate=args.lr_C,
                                  optimizer_name=args.optimizer_name,
                                  weight_decay=args.weight_decay,
                                  epsilon=args.epsilon,
                                  momentum=args.momentum
                                  )

    optimizer_F = build_optimizer(model=net_F,
                                  learning_rate=args.lr_F,
                                  optimizer_name=args.optimizer_name,
                                  weight_decay=args.weight_decay,
                                  epsilon=args.epsilon,
                                  momentum=args.momentum
                                  )

    best_psnr_C, best_psnr_F = 0., 0.
    for epoch in range(start_epoch, args.epochs):
        # ===== train =====
        adjust_learning_rate(optimizer_C, epoch, args.lr_C)
        adjust_learning_rate(optimizer_F, epoch, args.lr_F)
        logging.info('Epoch {} learning rate {} for C, {} for F'.format(epoch, optimizer_C.param_groups[0]['lr'], optimizer_F.param_groups[0]['lr']))

        batch_time = AverageMeter()
        batch_time_C = AverageMeter()
        batch_time_F = AverageMeter()
        losses_C = AverageMeter()
        losses_F = AverageMeter()

        # mutate model train / eval states
        if args.freeze_net_C:
            logging.info('Freezing coarse network.')
            net_C.train()
        else:
            logging.info('Training coarse network.')
            net_C.eval()

        if args.freeze_net_F:
            logging.info('Freezing fine network.')
            net_F.train()
        else:
            logging.info('Training fine network.')
            net_F.eval()

        end = time.time()
        for batch_idx, sample in enumerate(train_loader):

            in_videos_C, gt_videos_C = sample[0].to(rank), sample[1].to(rank)  # (b,c,d,w,h)
            batch_size, _, nf, _, _ = in_videos_C.shape

            # ===== optimize coarse network =====
            if args.freeze_net_C:
                with torch.no_grad():
                    model_out_C = net_C(in_videos_C)
                    loss_C = F.mse_loss(model_out_C, gt_videos_C)
            else:
                optimizer_C.zero_grad()

                model_out_C = net_C(in_videos_C)  # (b, c, d, h, w)
                # compute loss of coarse net and update
                loss_C = F.mse_loss(model_out_C, gt_videos_C)
                loss_C.backward()
                optimizer_C.step()

            losses_C.update(loss_C.item(), batch_size)
            # timing for coarse net
            c_end = time.time()
            batch_time_C.update(c_end - end)

            # ===== optimize fine network =====
            # take the frame in the middle as ground truth.
            # gt_frames_F = gt_videos_C[:, :, gt_videos_C.shape[2] // 2, :, :]
            #
            # if args.freeze_net_F:
            #     with torch.no_grad():
            #         model_out_F = net_F(model_out_C.detach(), in_videos_C)
            #         loss_F = F.mse_loss(model_out_F, gt_frames_F)
            # else:
            #     optimizer_F.zero_grad()
            #
            #     # compute loss of finenet and update
            #     model_out_F = net_F(model_out_C.detach(), in_videos_C)
            #     loss_F = F.mse_loss(model_out_F, gt_frames_F)
            #     loss_F.backward()
            #     optimizer_F.step()
            #
            # losses_F.update(loss_F.item(), batch_size)
            # # timing for fine net
            # batch_time_F.update(time.time() - c_end)
            #
            # # logistics
            # batch_time.update(time.time() - end)
            # end = time.time()

            global_step = len(train_loader) * epoch + batch_idx
            if batch_idx % 1 == 0:
                logging.info(('Epoch: [{0}][{1}/{2}]\t'
                              'iters: {3}\t'
                              'Time {batch_time_C.val:.3f} ({batch_time_C.sum:.3f})\t'
                              'C_net time: {batch_time_C.val:.3f} ({batch_time_C.sum:.3f})\t'
                              'F_net time: {batch_time_F.val:.3f} ({batch_time_F.sum:.3f})\t'
                              'Loss_C {loss_C.val:.4f} ({loss_C.avg:.4f})\t'
                              'Loss_F {loss_F.val:.4f} ({loss_F.avg:.4f})'
                              .format(epoch, batch_idx, len(train_loader), global_step,
                                      batch_time=batch_time, batch_time_C=batch_time_C, batch_time_F=batch_time_F,
                                      loss_C=losses_C, loss_F=losses_F)
                              ))

        # if (epoch + 1) % 1 == 0:
        #     save_checkpoint(net_C.state_dict(),
        #                     filename=args.checkpoint_dir_C + "checkpoints_small_" + str(epoch + 1) + ".pth.tar")
        #     save_checkpoint(net_F.state_dict(),
        #                     filename=args.checkpoint_dir_F + "checkpoints_small_" + str(epoch + 1) + ".pth.tar")
        #
        # # validate
        # if args.eval_on_the_fly and (epoch + 0) % args.eval_interval == 0:
        #     logging.info('Starting Evaluation on Epoch {}'.format(epoch))
        #     avg_psnr_C, avg_psnr_F = validate(args, net_C, net_F)
        #
        #     # save checkpoint with best PSNR
        #     if avg_psnr_C > best_psnr_C:
        #         best_psnr_C = avg_psnr_C
        #
        #         shutil.copyfile(src=args.checkpoint_dir_C + "checkpoints_small_" + str(epoch + 1) + ".pth.tar",
        #                         dst=args.checkpoint_dir_C + "checkpoints_best_" + str(epoch + 1) + ".pth.tar"
        #                         )
        #         logging.info('Best C PSNR {} found at {} epoch'.format(best_psnr_C, epoch))
        #
        #     if avg_psnr_F > best_psnr_F:
        #         best_psnr_F = avg_psnr_F
        #         shutil.copyfile(src=args.checkpoint_dir_F + "checkpoints_small_" + str(epoch + 1) + ".pth.tar",
        #                         dst=args.checkpoint_dir_F + "checkpoints_best_" + str(epoch + 1) + ".pth.tar"
        #                         )
        #         logging.info('Best F PSNR {} found at {} epoch'.format(best_psnr_F, epoch))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddp_main(rank, world_size):
    setup(rank, world_size)

    args = get_args('train')

    makedir(args.checkpoint_dir_C)
    makedir(args.checkpoint_dir_F)
    makedir(args.logdir)

    train(args, rank)
    cleanup()


def main():
    args = get_args('train')

    # Create folder
    makedir(args.checkpoint_dir_C)
    makedir(args.checkpoint_dir_F)
    makedir(args.logdir)

    train(args)


if __name__ == '__main__':
    world_size = 2

    mp.spawn(ddp_main,
             args=(world_size,),
             nprocs=2,
             join=True
             )

