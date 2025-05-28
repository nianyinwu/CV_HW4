""" Training a Image Restoration model """

import os
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import tqdm_bar, total_loss_func, compute_psnr
from eval import evaluation
from model import PromptIR
from dataloader import dataloader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# ignore warnings
warnings.filterwarnings('ignore')

# Enable fast training
cudnn.benchmark = True

def get_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        '--device',
        type=str,
        choices=[
            "cuda",
            "cpu"
        ],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='../datas',
        help='the path of input data'
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./saved_model',
        help='the path of save the training model'
    )
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=100,
        help='number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=32,
        help='batch size'
    )
    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=2e-4,
        help='learning rate'
    )

    return parser.parse_args()

def train(
    args: argparse.Namespace,
    cur_epoch: int,
    train_device: torch.device,
    train_model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    grad_scaler: torch.cuda.amp.GradScaler,
) -> torch.Tensor:
    """
    Train the model for one epoch

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        cur_epoch (int): Current training epoch.
        train_device (torch.device): Device to train on (CPU or GPU).
        train_model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        grad_scaler (GradScaler):  Automatic mixed-precision (AMP) gradient scaler
        scheduler: Learning rate scheduler

    Returns:
        Tuple[torch.Tensor, float]: The average training loss and accuracy.
    """
    train_model.train()

    total_loss = 0.0
    total_psnr = 0.0

    for images, targets in (pbar := tqdm(data_loader, ncols=120)):
        images = images.to(train_device)
        targets = targets.to(train_device)

        optimizer.zero_grad()

        with autocast():
            pred = train_model(images)
            loss = total_loss_func(pred, targets)
            psnr = compute_psnr(pred, targets)

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        total_psnr += psnr.item()
        total_loss += loss.item()

        tqdm_bar('Train', pbar, loss.detach().cpu(), cur_epoch, args.epochs)

    avg_psnr = total_psnr / len(data_loader)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, avg_psnr

if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(device)
    scaler = GradScaler()


    # Ensure the save path exist
    os.makedirs(opt.save_path, exist_ok=True)

    model = PromptIR().to(device)


    # Setting dataloader for training and validation
    train_loader = dataloader(args=opt, mode='train')
    val_loader = dataloader(args=opt, mode='valid')

    writer = SummaryWriter(log_dir=opt.save_path)

    # Setting the optimizer and scheduler
    optim_func = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_func, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )


    best_psnr = 0
    best_loss = float('inf')

    for epoch in range(opt.epochs):

        train_loss, train_psnr = train(
                                    opt,
                                    epoch,
                                    device,
                                    model,
                                    train_loader,
                                    optim_func,
                                    scaler
                                )

        val_loss, val_psnr = evaluation(device, model, val_loader)



        current_lr = optim_func.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1}/{opt.epochs} | "
            f"Train Loss: {train_loss:.4f}  | "
            f"Train PSNR: {train_psnr:.2f} dB  | "
            f"Valid Loss: {val_loss:.4f}    | "
            f"PSNR: {val_psnr:.2f} dB  | "
            f"LR: {current_lr:.1e}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_func.state_dict(),
                'psnr': val_psnr,
                'loss': val_loss
            }, os.path.join(opt.save_path, 'val_best.pth'))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_func.state_dict(),
                'psnr': val_psnr,
                'loss': val_loss
            }, os.path.join(opt.save_path, 'val_loss_best.pth'))

        lr_scheduler.step(val_loss)

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/PSNR", train_psnr, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/PSNR", val_psnr, epoch)


    torch.save(model.state_dict(), os.path.join(opt.save_path, 'last.pth'))

    writer.close()
