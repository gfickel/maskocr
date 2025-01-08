import math
import random
import time
import types

from tqdm import tqdm
import wandb
import schedulefree
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from fastai.callback.schedule import minimum, steep, valley, slide
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.optimizer import Adam
from fastcore.basics import store_attr

from utils import *
from vit_mae import ViT, MAE


device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(0)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return x + pe.to(x.device)

class LatentContextualRegressor(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_len):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, mask_queries, visible_patch_representations):
        # mask_queries shape: [num_masked_patches, batch_size, embed_dim]
        # Add positional encoding to mask_queries
        mask_queries = self.pos_encoder(mask_queries)
        
        # visible_patch_representations shape: [batch_size, num_patches, embed_dim]
        return self.transformer_decoder(mask_queries, visible_patch_representations)


class MaskOCR_Encoder(nn.Module):
    def __init__(self, img_height, img_width, patch_size, embed_dim, num_heads, num_layers, dropout, emb_dropout, overlap):
        super().__init__()
        self.vit = ViT(
            image_size=(img_height, img_width),
            patch_size=patch_size,
            num_classes=0,  # We don't need classification, so we use embed_dim as num_classes
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=embed_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            overlap=overlap,
        )
        self.num_patches = (img_height // patch_size[0]) * (img_width // patch_size[1])
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
    def forward(self, x, mask=None):
        x = self.vit(x)
        
        if mask is not None:
            x[mask] = 0
        
        return x

class MaskOCR_Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_sequence_length, dropout, emb_dropout):
        super().__init__()
        self.character_queries = nn.Parameter(torch.randn(max_sequence_length, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, max_sequence_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, memory, tgt_mask=None):
        # memory shape: [batch_size, num_patches, embed_dim]
        batch_size = memory.size(0)
        
        # Expand character queries to match batch size
        tgt = self.character_queries.expand(-1, batch_size, -1)
        # tgt shape: [max_sequence_length, batch_size, embed_dim]
        
        tgt = self.pos_encoder(tgt)
        # tgt shape remains: [max_sequence_length, batch_size, embed_dim]
        
        # Transpose memory to match expected input of transformer decoder
        memory = memory.transpose(0, 1)
        # memory shape: [num_patches, batch_size, embed_dim]
        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        # output shape: [max_sequence_length, batch_size, embed_dim]
        
        return output
    
class MaskOCR(nn.Module):
    def __init__(self, img_height, img_width, patch_size, embed_dim, num_heads, num_encoder_layers, num_decoder_layers,
                 vocab_size, max_sequence_length, dropout, emb_dropout, overlap):
        super().__init__()
        self.encoder = MaskOCR_Encoder(img_height, img_width, patch_size, embed_dim, num_heads, num_encoder_layers, dropout, emb_dropout, overlap)
        self.decoder = MaskOCR_Decoder(embed_dim, num_heads, num_decoder_layers, max_sequence_length, dropout, emb_dropout)
        
        self.classifier = nn.Linear(embed_dim, vocab_size)
        
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        
        # For visual pre-training
        self.mae = MAE(
            encoder=self.encoder.vit,
            decoder_dim=embed_dim,
            masking_ratio=0.7,
            decoder_depth=4,
            decoder_heads=num_heads,
            decoder_dim_head=64
        )
        
        self.regressor = LatentContextualRegressor(embed_dim, num_heads, 4, max_sequence_length)
    
    def forward(self, images, mask=None):
        encoder_output = self.encoder(images, mask)
        decoder_output = self.decoder(encoder_output)
        decoder_output = decoder_output.transpose(0, 1)
        logits = self.classifier(decoder_output)
        logits = logits.view(-1, self.max_sequence_length, self.vocab_size)
        return logits
    
    def visual_pretraining_forward(self, images):
        return self.mae(images)
    
    def language_pretraining_forward(self, images, char_mask, patch_mask):
        # images shape: [batch_size, 3, img_height, img_width]
        # char_mask shape: [batch_size, max_sequence_length]
        # patch_mask shape: [batch_size, num_patches]
        
        with torch.no_grad():
            # encoder_output shape: [batch_size, num_patches, embed_dim]
            encoder_output = self.encoder(images, patch_mask)
        
        # masked_encoder_output shape: [batch_size, num_patches, embed_dim]
        masked_encoder_output = encoder_output.clone()
        masked_encoder_output[patch_mask] = 0
        
        # decoder_output shape: [max_sequence_length, batch_size, embed_dim]
        decoder_output = self.decoder(masked_encoder_output)
        
        # classifier output shape: [max_sequence_length, batch_size, vocab_size]
        return self.classifier(decoder_output), char_mask



def base_train_loop(project, model, train_function, train_dataloader, val_dataloader, device, vocab,
                    model_name=None, num_epochs=40, use_wandb=False, config=None,
                    start_epoch=0, temp_model_path=None, version='', start_lr=1e-4, min_lr=1e-5, 
                    plateau_threshold=-1, use_schedulefree=False, weight_decay=0.05, training_loss_fn=None, fastai_loss_fn=None, **kwargs):
    """
    Base training loop that handles the common training functionality for all model types.
    This function implements the core training loop with configurable loss computation.
    
    The loss computation can be handled in two ways:
    1. Traditional PyTorch style (outputs, targets) for standard training
    2. Custom computation that handles both forward pass and loss for special cases
    """
    use_schedulefree = False
    model.to(device)

    if start_lr < 0:
        dls = DataLoaders(train_dataloader, val_dataloader)

        learn = Learner(
            dls,
            model,
            loss_func=fastai_loss_fn,
            opt_func=Adam,
            lr=1e-3,
        )
        lr_min, lr_steep, lr_valley, lr_slide  = learn.lr_find(start_lr=1e-4, end_lr=1e-1, num_it=100, suggest_funcs=(minimum, steep, valley, slide))
        start_lr = lr_valley
        plt.savefig(f'lr_finder_{model_name}.png', dpi=300)
        print(f"Found suggested learning rate: start {start_lr}, valley {lr_valley}, min {lr_min} {lr_steep}")

        after_fastai = kwargs.get('after_fastai', None)
        if after_fastai is not None:
            after_fastai(model)
    curr_lr = start_lr
    
    # Initialize optimizer
    if use_schedulefree:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=start_lr, 
                                                 weight_decay=weight_decay, betas=(.9, .95))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=start_lr, 
                              weight_decay=weight_decay, betas=(.9, .95))
    
    # Initialize learning rate scheduler if needed
    scheduler = OneCycleLR(optimizer, max_lr=start_lr, total_steps=len(train_dataloader)*num_epochs)
    
    # Load optimizer and scheduler states if resuming
    if start_epoch > 0 and temp_model_path:
        checkpoint = torch.load(temp_model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if plateau_threshold < 0 and not use_schedulefree:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Apply model-specific setup (like encoder freezing)
    freeze_encoder = kwargs.get('freeze_encoder', False)
    for param in model.encoder.parameters():
        param.requires_grad = not freeze_encoder
            
    loss_history = []
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        if use_schedulefree: optimizer.train()
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            loss = training_loss_fn(model, batch, device)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
            if use_wandb:
                wandb.log({
                    "epoch": epoch + (batch_idx+1)/len(train_dataloader),
                    "train_loss": loss_history[-1],
                    "learning_rate": curr_lr,
                })
            
        
        avg_loss = np.mean(loss_history[-len(train_dataloader):])
        
        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_loss = training_loss_fn(model, val_batch, device)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Learning rate scheduling
        
        if plateau_threshold > 0:
            if is_in_plateau(loss_history, threshold=plateau_threshold):
                # Different reduction factors for different training types
                reduction_factor = 0.5 if kwargs.get('is_text_recognition', False) else 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= reduction_factor
                    curr_lr = param_group['lr']
                loss_history = []
                print(f"{version} - Learning rate reduced to {curr_lr}")
        
        # Logging
        print(f"{version} - Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"lr: {curr_lr:.6f}, "
              f"Time: {time.time() - epoch_start_time:.2f} seconds")
              
        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "val_loss": avg_val_loss,
                "epoch_time": time.time() - epoch_start_time,
            })
        
        # Save checkpoint
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if plateau_threshold < 0 and not use_schedulefree:
            state_dict['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(state_dict, temp_model_path)


def train_text_recognition(project, model, train_function, train_dataloader, val_dataloader, device, vocab, 
                         model_name=None, num_epochs=40, use_wandb=False, config=None,
                         freeze_encoder=False, start_lr=1e-4, min_lr=1e-5, plateau_threshold=-1, 
                         use_schedulefree=False, version='', weight_decay=0.05, **kwargs):
    """
    Specialized training function for text recognition.
    Uses a standard PyTorch-style loss computation (outputs, targets).
    """
    def compute_text_recognition_loss(model, batch, device):
        images, text_indices = batch
        images = images.to(device)
        text_indices = text_indices.to(device)
        outputs = model(images)
        return text_recognition_loss(outputs, text_indices, padding_idx=-1)
    
    def fastai_loss_fn(pred, target):
        return text_recognition_loss(pred, target, padding_idx=-1)

    return base_train_loop(
        project, model, train_function, train_dataloader, val_dataloader, device, vocab,
        model_name=model_name, num_epochs=num_epochs, use_wandb=use_wandb, config=config,
        training_loss_fn=compute_text_recognition_loss,
        fastai_loss_fn=fastai_loss_fn,
        freeze_encoder=freeze_encoder,
        start_lr=start_lr,
        min_lr=min_lr,
        plateau_threshold=plateau_threshold,
        use_schedulefree=use_schedulefree,
        version=version,
        is_text_recognition=True,
        weight_decay=weight_decay,
        **kwargs
    )

def train_visual_pretraining(project, model, train_function, train_dataloader, val_dataloader, device, vocab,
                           model_name=None, num_epochs=40, use_wandb=False, config=None,
                           start_lr=1e-4, min_lr=1e-5, plateau_threshold=-1, 
                           use_schedulefree=False, version='', **kwargs):
    """
    Training function for visual pretraining that provides both training and fastai loss functions.
    """
    # Regular training loss
    def training_loss_fn(model, batch, device):
        images, _ = batch
        images = images.to(device)
        _, loss, _, _ = model.visual_pretraining_forward(images)
        return loss

    # Fastai loss function for learning rate finding
    def fastai_forward(self, x):
        x = x.to(device)
        _, loss, _, _ = self.visual_pretraining_forward(x)
        return loss
    
    def fastai_loss_fn(pred, target):
        return pred
    
    # Store and set up forward method for fastai
    original_forward = model.forward
    model.forward = types.MethodType(fastai_forward, model)
    
    def fix_model(model):
        model.forward = original_forward

    try:
        res = base_train_loop(
            project, model, train_function, train_dataloader, val_dataloader, device, vocab,
            model_name=model_name, num_epochs=num_epochs, use_wandb=use_wandb, config=config,
            training_loss_fn=training_loss_fn,
            fastai_loss_fn=fastai_loss_fn,
            start_lr=start_lr,
            min_lr=min_lr,
            plateau_threshold=plateau_threshold,
            use_schedulefree=use_schedulefree,
            version=version,
            after_fastai=fix_model,
            **kwargs
        )
    finally:
        model.forward = original_forward

    return res


def compute_text_recognition_loss(model, batch):
    """Computes the text recognition loss for a given batch."""
    images, text_indices = batch
    images = images.to(device)
    text_indices = text_indices.to(device)
    outputs = model(images)
    return text_recognition_loss(outputs, text_indices, padding_idx=-1)

def compute_visual_pretraining_loss(model, batch):
    """Computes the visual pretraining loss for a given batch."""
    print(batch)
    images, _ = batch
    images = images.to(device)
    _, loss, _, _ = model.visual_pretraining_forward(images)
    return loss


def text_recognition_loss(predictions, targets, padding_idx):
    # predictions shape: [batch_size, max_sequence_length, vocab_size]
    # targets shape: [batch_size, max_sequence_length]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
    
    # Reshape predictions and targets for loss calculation
    predictions = predictions.view(-1, predictions.size(-1))
    targets = targets.view(-1)
    
    return loss_fn(predictions, targets)

def visual_pretraining_loss(decoded_patches, true_patch_values, predicted_masked_representations, true_masked_representations, lambda_value=0.05):
    # decoded_patches shape: [num_masked_patches, batch_size, patch_dim]
    # true_patch_values shape: [num_masked_patches, batch_size, patch_dim]
    # predicted_masked_representations shape: [num_masked_patches, batch_size, embed_dim]
    # true_masked_representations shape: [num_masked_patches, batch_size, embed_dim]
    
    prediction_loss = nn.MSELoss()(decoded_patches, true_patch_values)
    alignment_loss = nn.MSELoss()(predicted_masked_representations, true_masked_representations)
    return prediction_loss + lambda_value * alignment_loss


def create_vertical_strip_mask(batch_size, num_patches, mask_ratio):
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    num_masked_patches = int(num_patches * mask_ratio)
    for i in range(batch_size):
        masked_indices = torch.randperm(num_patches)[:num_masked_patches]
        mask[i, masked_indices] = True
    return mask

def create_char_and_patch_masks(batch_size, num_patches, num_chars, mask_ratio=0.15):
    char_mask = torch.rand(batch_size, num_chars) < mask_ratio
    patch_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    
    for i in range(batch_size):
        masked_chars = char_mask[i].nonzero().squeeze(1)  # Change: squeeze(1) instead of squeeze()
        
        # If no characters are masked, randomly mask one character
        if masked_chars.numel() == 0:
            masked_chars = torch.tensor([random.randint(0, num_chars - 1)])
            char_mask[i, masked_chars] = True
        
        for char in masked_chars:
            start_patch = (char * num_patches) // num_chars
            end_patch = ((char + 1) * num_patches) // num_chars
            patch_mask[i, start_patch:end_patch] = True
    
    return char_mask, patch_mask

def language_pretraining_loss(predictions, targets, char_mask):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(predictions.view(-1, predictions.size(-1)), targets.view(-1))
    masked_losses = losses.view_as(targets)[char_mask]
    return masked_losses.mean()

def train_language_pretraining(model, dataloader, device, vocab, num_epochs=50):
    model.to(device)
    
    # Freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        for images, text_indices in dataloader:
            images = images.to(device)
            text_indices = text_indices.to(device)
            
            char_mask, patch_mask = create_char_and_patch_masks(images.size(0), model.encoder.num_patches, text_indices.size(1))
            char_mask, patch_mask = char_mask.to(device), patch_mask.to(device)
            
            optimizer.zero_grad()
            
            predictions, char_mask = model.language_pretraining_forward(images, char_mask, patch_mask)
            loss = language_pretraining_loss(predictions, text_indices, char_mask)
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
