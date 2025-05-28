""" Evaluate a Image Restoration model """

import os
import argparse
import warnings

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from model import PromptIR
from dataloader import dataloader
from utils import tqdm_bar, total_loss_func, compute_psnr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# ignore warnings
warnings.filterwarnings('ignore')


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
        '--weights',
        '-w',
        type=str,
        default='./saved_model/val_best.pth',
        help='the path of input model weights'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='batch size'
    )

    return parser.parse_args()


def evaluation(
    device: torch.device,
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, float]:
    """
    Evaluate the model on the validation dataset.

    Args :
        device : Evaluation device.
        model : Trained model to evaluate.
        valid_loader : DataLoader for the validation set.
        gt_json : The path of ground truth annotation file.

    Returns:
        avg_loss : average loss over validation set
        avg_psnr : average PSNR over validation set
    """

    # Evaluation PSNR
    model.eval()
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for images, targets in (pbar := tqdm(valid_loader, ncols=120)):
            images = images.to(device)
            targets = targets.to(device)

            with autocast():
                pred = model(images)
                loss = total_loss_func(pred, targets)
                psnr = compute_psnr(pred, targets)

            total_loss += loss.item()
            total_psnr += psnr
            tqdm_bar('Evaluation', pbar)

    avg_loss = total_loss / len(valid_loader)
    avg_psnr = total_psnr / len(valid_loader)

    return avg_loss, avg_psnr



def main():
    """
    Main function to evaluate the model.
    """
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PromptIR().to(device)
    checkpoint = torch.load(opt.weights, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    val_loader = dataloader(args=opt, mode='valid')

    # Evaluate
    loss, psnr = evaluation(device, model, val_loader)

    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
