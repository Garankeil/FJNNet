import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from dataset.dataset import ScatteringDataset
from utils.utils import setup_seed, sec_to_hms, ssim, psnr, save_image
from network.FJNNET import FJNNet


def parse_args():
    parser = argparse.ArgumentParser(description='Training and Evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--benchmark', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--train-root', type=str, default='./data/train')
    parser.add_argument('--eval-root', type=str, default='./data/eval')
    parser.add_argument('--save-path', type=str, default='./checkpoints')
    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model.pt')

    return parser.parse_args()


def validate_and_monitor(epoch, writer, loader, model, optimizer, device, save_path, best_metric=0, metric='ssim'):
    model.eval()
    metric_values = []
    with torch.no_grad():
        step = 1
        for sp, label in loader:
            sp, label = sp.to(device), label.to(device)
            output = model(sp)
            if writer:
                writer.add_images(f'Validation_Epoch_{epoch + 1}', output, step)

            val = ssim(output, label) if metric == 'ssim' else psnr(output, label)
            metric_values.append(val.item())
            step += 1

    avg_metric = np.mean(metric_values)
    print(f'Epoch {epoch + 1} - {metric.upper()}: {avg_metric:.6f} (Best: {best_metric:.6f})')

    if avg_metric > best_metric:
        best_metric = avg_metric
        os.makedirs(save_path, exist_ok=True)
        checkpoint = {
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_path, 'best_model.pt'))

    model.train()
    return best_metric


def train(model, epochs, optimizer, scheduler, train_loader, eval_loader, writer, scaler, rank, save_path):
    total_start_time = time.time()
    loss_mse = nn.MSELoss().to(rank)
    best_ssim = 0

    if rank == 0:
        print("Starting training process...")

    for i in range(epochs):
        epoch_start_time = time.time()
        train_loader.sampler.set_epoch(i)
        loss_disp = 0

        for sp, label in train_loader:
            sp, label = sp.to(rank), label.to(rank)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(sp)
                loss = loss_mse(output, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_disp += loss.item()

        scheduler.step()

        if rank == 0:
            avg_loss = loss_disp / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - total_start_time
            pred_remain_time = sec_to_hms(epoch_time * (epochs - i))

            print(f'Epoch [{i + 1}/{epochs}] Loss: {avg_loss:.8f} | '
                  f'Elapsed: {sec_to_hms(total_time)} | Remaining: {pred_remain_time}')

            best_ssim = validate_and_monitor(i, writer, eval_loader, model, optimizer, rank, save_path, best_ssim,
                                             'ssim')
            writer.add_scalar('Loss/train', avg_loss, i)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], i)

    dist.barrier()


def validation(rank, model, loader, test_path, recon_folder, recon=False, ssim_only=False):
    model.eval()
    ssim_list, psnr_list = [], []
    with torch.no_grad():
        step = 0
        for sp, label in loader:
            sp, label = sp.to(rank), label.to(rank)
            output = model(sp)
            if ssim_only:
                ssim_list.append(ssim(output, label).item())
                psnr_list.append(psnr(output, label).item())
            if recon:
                save_image(output, os.path.join(test_path, recon_folder), step + 1)
            step += 1

    if ssim_only and rank == 0:
        print(f'Test Results - SSIM: {np.mean(ssim_list):.6f}, PSNR: {np.mean(psnr_list):.6f}')


def main():
    args = parse_args()
    setup_seed(args.seed)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cudnn.benchmark = args.benchmark

    model = FJNNet().to(rank)

    if args.train:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        train_set = ScatteringDataset(args.train_root, 'speckle', 'label')
        eval_set = ScatteringDataset(args.eval_root, 'speckle', 'label')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set),
                                  num_workers=args.num_workers)
        eval_loader = DataLoader(eval_set, batch_size=1, num_workers=args.num_workers)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler()
        writer = SummaryWriter('./logs') if rank == 0 else None

        train(model, args.epochs, optimizer, scheduler, train_loader, eval_loader, writer, scaler, rank, args.save_path)
    else:
        # Evaluate Only
        if rank == 0:
            print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=f'cuda:{rank}')
        model.load_state_dict(checkpoint['model'])

        test_set = ScatteringDataset(args.eval_root, 'speckle', 'label')
        test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers)
        validation(rank, model, test_loader, './results', 'recon', recon=True, ssim_only=True)


if __name__ == '__main__':
    main()
