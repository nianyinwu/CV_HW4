""" Inference script for Image Restoration Model """

import argparse
import warnings
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from utils import tqdm_bar

import torch
from model import PromptIR
from dataloader import dataloader

# ignore warnings
warnings.filterwarnings('ignore')

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument(
        '--device',
        type=str,
        choices=["cuda", "cpu"],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='../datas',
        help='Path to input data'
    )
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        default='./best.pth',
        help='Path to model weights'
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./saved_model',
        help='the path of save the training model'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    return parser.parse_args()

def test(
    args: argparse.Namespace,
    test_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> List[Tuple[str, str]]:
    """
    Perform inference on the test set.
    """

    results = []
    with torch.no_grad():
        for fname, image in (pbar := tqdm(data_loader, ncols=120)):
            fname = fname[0]
            image = image.to(args.device)
            clean_image = test_model(image)
            clean_image = clean_image[0]

            results.append((fname, clean_image))
            tqdm_bar('Test', pbar)

    return results


def img2npz(save_path: str, predictions: List[Tuple[str, torch.Tensor]]) -> None:
    """
    Transform images to numpy arrays and save them as .npz files.
    """

    images_dict = {}
    predictions = sorted(predictions, key=lambda x: x[0])
    for fname, clean_img in predictions:
        # Convert tensor to numpy array
        clean_img = clean_img.clamp(0, 1).mul(255).byte().cpu().numpy()
        images_dict[fname] = clean_img

    # Save to .npz file
    np.savez(f"{save_path}/pred.npz", **images_dict)

if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    test_loader = dataloader(opt, 'test')

    # Load model
    model = PromptIR().to(device)
    checkpoint = torch.load(opt.weights, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference
    pred = test(opt, model, test_loader)

    img2npz(opt.save_path, pred)

    print(f"Saved predictions to {opt.save_path}/pred.npz")
