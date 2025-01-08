import os
import argparse
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import wandb

from utils import *
from dataloader import ALPRDataset, create_ocr_transform
from maskocr import *

# Set up interactive mode
plt.ion()



def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for image processing and model configuration.")
    
    parser.add_argument('--dataset_path', type=str, default='../', help='Base directory containing the datasets')
    parser.add_argument('--ds_frac', type=float, default=1.0, help='Dataset Fraction to use')
    parser.add_argument('--img_height', type=int, default=32, help='Height of the input image')
    parser.add_argument('--img_width', type=int, default=128, help='Width of the input image')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[32, 8], help='Patch size for image processing (height, width)')
    parser.add_argument('--batch_size', type=int, default=512*7, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=120, help='Number of text recognition training epochs')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--max_sequence_length', type=int, default=7, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--emb_dropout', type=float, default=0.1, help='Embedding dropout rate')
    parser.add_argument('--version', type=str, required=True, help='Training Version')
    parser.add_argument('--norm_image', type=int, default=0, help='Normalize the input image')
    parser.add_argument('--overlap', type=int, default=0, help='Patch Overlap')
    parser.add_argument('--device', type=int, default=0, help='Normalize the input image')
    parser.add_argument('--start_lr', type=float, default=1e-4, help='Starting learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Starting learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight Decay')
    parser.add_argument('--aug_strength', type=float, default=1.0, help='Augmentation strength')
    parser.add_argument('--plateau_thr', type=int, default=-1, help='Number of batches to use on dlib plateau detection')
    parser.add_argument('--wandb', action='store_true', help='Log with wandb or not')
    parser.add_argument('--schedulefree', action='store_true', help='Use FAIR ScheduleFree')
    
    args = parser.parse_args()
    return args

def save_arguments_to_json(args, filename):
    # Convert args namespace to dictionary
    args_dict = vars(args)
    
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    print(f"Arguments saved to {filename}")

def main():
    cfg = parse_arguments()
    save_arguments_to_json(cfg, os.path.join('configs', f'{cfg.version}.json'))
    current_device = cfg.device
    device = torch.device(f'cuda:{current_device}' if torch.cuda.is_available() else 'cpu')
    
    vocab = FULL_ALPHABET
    vocab_size = len(vocab)+1

    model = MaskOCR(cfg.img_height, cfg.img_width, cfg.patch_size, cfg.embed_dim, cfg.num_heads, cfg.num_encoder_layers,
                    cfg.num_decoder_layers, vocab_size, cfg.max_sequence_length, dropout=cfg.dropout, emb_dropout=cfg.emb_dropout,
                    overlap=cfg.overlap)

    val_data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data_transform = create_ocr_transform(augment_strength=cfg.aug_strength)

    if cfg.img_height == 32:
        train_ds = 'output_images_plates_gt'
        val_ds = 'plates_ccpd_weather'
    elif cfg.img_height == 48:
        train_ds = 'plates_ccpd_base_48'
        val_ds = 'plates_ccpd_weather_48'

    train_dataset = ALPRDataset(
        os.path.join(cfg.dataset_path, train_ds, 'alpr_annotation.csv'),
        os.path.join(cfg.dataset_path, train_ds),
        train_data_transform, ds_frac=cfg.ds_frac)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=2, drop_last=True, prefetch_factor=32)

    val_dataset = ALPRDataset(
        os.path.join(cfg.dataset_path, val_ds, 'alpr_annotation.csv'),
        os.path.join(cfg.dataset_path, val_ds),
        val_data_transform, ds_frac=cfg.ds_frac)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=1, drop_last=False)

    # summary(model, input_size=(2, 3, cfg.img_height, cfg.img_width), depth=5)

    train_model(f'maskocr', model, train_visual_pretraining, train_dataloader, val_dataloader, device, vocab,
                model_name=f'train_visual_pretraining_{cfg.version}', num_epochs=max(1,cfg.epochs//4), version=cfg.version,
                start_lr=cfg.start_lr, min_lr=cfg.min_lr, plateau_threshold=cfg.plateau_thr, use_wandb=cfg.wandb,
                use_schedulefree=cfg.schedulefree, config=cfg)
    
    # Then, train for text recognition
    train_model(f'maskocr', model, train_text_recognition, train_dataloader, val_dataloader, device, vocab,
                model_name=f'train_text_recognition_freeze_{cfg.version}', num_epochs=max(1,cfg.epochs//2), freeze_encoder=True,
                version=cfg.version, start_lr=cfg.start_lr, min_lr=cfg.min_lr, plateau_threshold=cfg.plateau_thr, use_wandb=cfg.wandb,
                use_schedulefree=cfg.schedulefree, weight_decay=0, config=cfg)
    train_model(f'maskocr', model, train_text_recognition, train_dataloader, val_dataloader, device, vocab,
                model_name=f'train_text_recognition_full_1_{cfg.version}', num_epochs=max(1,cfg.epochs), freeze_encoder=False,
                version=cfg.version, start_lr=cfg.start_lr, min_lr=cfg.min_lr, plateau_threshold=cfg.plateau_thr, use_wandb=cfg.wandb,
                use_schedulefree=cfg.schedulefree, weight_decay=cfg.weight_decay, config=cfg)
    torch.save(model.state_dict(), f'model_bin/my_model_{cfg.version}.pth')

if __name__ == "__main__":
    main()