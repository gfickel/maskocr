import os

import wandb
import numpy as np
import torch
import Levenshtein as lev
from matplotlib import pyplot as plt


PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z']
ADS =       ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
             'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

FULL_ALPHABET = PROVINCES+ADS
PROVINCES_IDX = [x+1 for x in range(len(PROVINCES))]
ADS_IDX = [x+len(PROVINCES)+1 for x in range(len(ADS))]

def train_model(project, model, train_function, train_dataloader, val_dataloader, device, vocab, 
                model_name=None, num_epochs=40, use_wandb=False, config=None, **train_kwargs):
    """
    Main training controller that manages model loading, training, and saving.
    This function handles the high-level training workflow, including checkpoint management.
    
    Args:
        project: Project name for wandb logging
        model: The model to be trained
        train_function: The specific training function to use (e.g., train_text_recognition)
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to run the model on
        vocab: Vocabulary for text recognition
        model_name: Name for saving the model (defaults to train_function name)
        num_epochs: Number of epochs to train
        use_wandb: Whether to use Weights & Biases logging
        config: Configuration for wandb
        **train_kwargs: Additional arguments passed to the training function
    """
    if model_name is None:
        model_name = train_function.__name__

    final_model_path = os.path.join('model_bin', f"{model_name}_final.pth")
    temp_model_path = os.path.join('model_bin', f"{model_name}_temp.pth")
    
    # Load existing checkpoint if available
    start_epoch = 0
    if os.path.exists(temp_model_path):
        print(f"Loading existing temporary model from {temp_model_path}")
        checkpoint = torch.load(temp_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    elif os.path.exists(final_model_path):
        print(f"Loading existing final model from {final_model_path}")
        checkpoint = torch.load(final_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return
    else:
        print(f"No existing model found. Training new model using {model_name} for {num_epochs} epochs.")

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project=project, config=config, group=train_kwargs.get('version', ''), name=model_name)
        wandb.watch(model, log_freq=5)

    try:
        # Train the model
        train_function(project, model, train_function, train_dataloader, val_dataloader, device, vocab, 
                      num_epochs=num_epochs, start_epoch=start_epoch, model_name=model_name,
                      temp_model_path=temp_model_path, use_wandb=use_wandb, **train_kwargs)

        # Save the final model
        print(f"Saving final trained model to {final_model_path}")
        torch.save({'model_state_dict': model.state_dict()}, final_model_path)

        # Remove temporary checkpoint
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    finally:
        if use_wandb:
            wandb.finish()

    return model

def indices_to_text(indices, vocab):
    return ''.join([vocab[idx.item()-1] for idx in indices if idx.item() < len(vocab)])

def plot_reconstructed_images(fig, axes, model, dataloader, device, num_examples=3, mask_ratio=0.3):
    model.eval()
    if fig is None:
        fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4*num_examples))
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Get reconstructed patches and masked indices
            reconstructed_patches, _, masked_indices, masked_patches = model.visual_pretraining_forward(images)
            
            # Get the original patches
            original_patches = model.mae.to_patch(images)
            
            for i in range(min(num_examples, batch_size)):
                # Create a copy of the original patches
                reconstructed_image = original_patches[i].clone()
                
                # Replace masked patches with reconstructed ones
                reconstructed_image[masked_indices[i]] = reconstructed_patches[i]
                
                # Reshape back to image dimensions
                reconstructed_image = reconstructed_image.reshape(16, 32, 8, 3)
                reconstructed_image = reconstructed_image.permute(1, 0, 2, 3).reshape(32, 128, 3)
                
                # Clip the values to [0, 1] range
                reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
                
                # Plot original and reconstructed images
                axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().clamp(0, 1))
                axes[i, 0].set_title("Original")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(reconstructed_image.cpu())
                axes[i, 1].set_title("Reconstructed")
                axes[i, 1].axis('off')
            
            
            break  # We only need one batch
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1)

    # Save the plot as an image
    os.makedirs('plot_output', exist_ok=True)
    save_file = os.path.join('plot_output', f'reconstruct_{len(os.listdir("plot_output"))}.png')
    plt.savefig(save_file)

    return fig, axes

def plot_examples(fig, axes, model, dataloader, device, vocab, num_examples=3):
    model.eval()
    if fig is None:
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4*num_examples))
    
    for ax in axes.flatten():
        ax.clear()
        
    with torch.no_grad():
        for i, (images, text_indices) in enumerate(dataloader):
            if i >= num_examples:
                break
            
            images = images.to(device)
            text_indices = text_indices.to(device)
            
            # Get model predictions
            logits = model(images)
            _, predicted = torch.max(logits, 2)
            
            # Convert indices to text
            true_text = indices_to_text(text_indices[0], vocab)
            pred_text = indices_to_text(predicted[0], vocab)
            
            # Plot original image
            original_image = images[0].cpu().permute(1, 2, 0)
            axes[i, 0].imshow(np.clip(original_image/original_image.max(), 0, 1))
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            # Plot text comparison
            axes[i, 1].text(0.1, 0.6, f"Ground Truth: {true_text}", fontsize=12)
            axes[i, 1].text(0.1, 0.4, f"Prediction: {pred_text}", fontsize=12)
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1)

    # Save the plot as an image
    os.makedirs('plot_output', exist_ok=True)
    save_file = os.path.join('plot_output', f'ocr_{len(os.listdir("plot_output"))}.png')
    plt.savefig(save_file)
    
    return fig, axes


def decode_ocr(ocrs_logits):
    log_ocrs = torch.nn.functional.log_softmax(ocrs_logits, dim=2)
    
    def _reconstruct(labels, blank=0):
        new_labels = []
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # delete blank
        return [l for l in new_labels if l != blank]

    log_ocrs = log_ocrs.permute(1, 2, 0)
    ocrs_argmax = torch.argmax(log_ocrs, axis=1).tolist()
    return [_reconstruct(x) for x in ocrs_argmax]


def calculate_cer(pred_texts, target_texts):
    total_distance = 0
    total_characters = 0
    
    for pred, target in zip(pred_texts, target_texts):
        total_distance += lev.distance(target, pred)
        total_characters += len(target)
    
    cer = total_distance / total_characters if total_characters > 0 else float('inf')
    return cer

def calculate_wer(pred_texts, target_texts):
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(pred_texts, target_texts):
        pred_words = pred.split()
        target_words = target.split()
        total_distance += lev.distance(' '.join(target_words), ' '.join(pred_words))
        total_words += len(target_words)
    
    wer = total_distance / total_words if total_words > 0 else float('inf')
    return wer

def calculate_metrics(decoded_preds, target_ocrs_decoded):
    pred_texts = [''.join([FULL_ALPHABET[idx-1] for idx in pred]) for pred in decoded_preds]
    target_texts = [''.join([FULL_ALPHABET[idx-1] for idx in target]) for target in target_ocrs_decoded]
    
    cer = calculate_cer(pred_texts, target_texts)
    wer = calculate_wer(pred_texts, target_texts)
    
    return cer, wer
