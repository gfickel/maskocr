import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageEnhance, ImageFilter

from utils import *


class ALPRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, detection=False, return_image_path=False, resize=None, grayscale=False, ds_frac=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.return_image_path = return_image_path
        self.to_tensor = transforms.ToTensor()
        self.resize = resize
        self.grayscale = grayscale
        self.detection = detection
        self.spatial_transform = ALPRSpatialTransform(
            scale=(0.9, 3.5),
            input_size=224
        )
        if ds_frac:
            self.img_labels = self.img_labels.sample(frac=ds_frac, random_state=42)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = None
        while image is None:
            img_path = os.path.join(self.img_dir, os.path.basename(self.img_labels.iloc[idx, 0]))
            try:
                image = Image.open(img_path)
                if self.grayscale:
                    image = image.convert('L')
            except:
                image = None
            if image != None:
                break
            idx = (idx+1) % len(self.img_labels)
                    
        W, H = image.size
        # H, W = image.shape[1:]
        w_scale = 1#224/W
        h_scale = 1#224/H
        l,t,r,b = self.img_labels.iloc[idx, 1:5].tolist()
        kps = self.img_labels.iloc[idx, 5:5+8].tolist()
        
        ocrs = self.img_labels.iloc[idx, 5+8:].tolist()
        ocrs = [int(x) for x in ocrs]
        ocrs[0] = PROVINCES_IDX[ocrs[0]-1]
        try:
            ocrs[1:] = [ADS_IDX[x-1] for x in ocrs[1:]]
        except:
            print(ocrs, '\n', len(ADS_IDX))
            exit(1)

        gt_ocrs = torch.LongTensor(ocrs)

        gt_bboxes = torch.Tensor(
            [[l*w_scale, t*h_scale, r*w_scale, b*h_scale]])
        gt_kps = torch.Tensor(
            [[pt*w_scale if idx%2==0 else pt*h_scale for idx,pt in enumerate(kps)]]
        )
        gt_labels = torch.LongTensor([0])

        if self.transform:
            if self.detection:
                image, gt_bboxes, gt_kps = self.spatial_transform(image, gt_bboxes, gt_kps)
            image = self.transform(image)
        else:
            if self.resize:
                image = image.resize(self.resize)
                
            # image = self.to_tensor(image)
        
        if self.return_image_path:
            if self.detection:
                return image, img_path, gt_bboxes, gt_labels, gt_kps, gt_ocrs
            else:
                return image, img_path, gt_ocrs
        else:
            if self.detection:
                return image, gt_bboxes, gt_labels, gt_kps, gt_ocrs
            else:
                return image, gt_ocrs


class ALPRSpatialTransform:
    def __init__(self, 
                 input_size=224, 
                 scale=(0.8, 1.2),
                 interpolation=InterpolationMode.BILINEAR):
        self.input_size = input_size
        self.scale = scale
        self.interpolation = interpolation

    def calculate_valid_scale_range(self, bbox_width, bbox_height, image_width, image_height):
        """Calculate valid scale range that ensures bbox fits in final image"""
        # Maximum scale is limited by how much we can zoom in while keeping the bbox visible
        max_scale_w = image_width / bbox_width
        max_scale_h = image_height / bbox_height
        max_scale = min(max_scale_w, max_scale_h)
        
        # Minimum scale is determined by our scale parameter
        min_scale = self.scale[0]
        
        # Clamp max_scale to our scale parameter
        max_scale = min(max_scale, self.scale[1])
        
        return min_scale, max_scale

    def calculate_valid_crop_range(self, bbox, kps, crop_size, new_w, new_h):
        """Calculate valid crop range that ensures annotations remain in frame"""
        # Get bounds of all annotations
        points = []
        if bbox is not None:
            points.extend([(bbox[0].item(), bbox[1].item()),
                         (bbox[2].item(), bbox[3].item())])
        if kps is not None:
            kps_reshape = kps.view(-1, 2)
            points.extend([(x.item(), y.item()) for x, y in kps_reshape])
            
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        
        # Calculate valid crop ranges that keep all points in frame
        valid_left = max(0, min(new_w - crop_size, max_x - crop_size))
        valid_right = max(0, min(new_w - crop_size, min_x))
        valid_top = max(0, min(new_h - crop_size, max_y - crop_size))
        valid_bottom = max(0, min(new_h - crop_size, min_y))
        
        return valid_left, valid_right, valid_top, valid_bottom

    def __call__(self, image, bboxes=None, kps=None):
        # Get original size
        w, h = image.size
        
        # Calculate initial resize to maintain aspect ratio
        # if w >= h:
        #     new_h = self.input_size
        #     new_w = int(new_h * (w / h))
        # else:
        #     new_w = self.input_size
        #     new_h = int(new_w * (h / w))
        ratio = min(self.input_size/w, self.input_size/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # First resize to maintain aspect ratio
        image = TF.resize(image, (new_h, new_w), self.interpolation)
        
        # Calculate scaling ratio for coordinates
        w_scale = new_w / w
        h_scale = new_h / h
        
        # Update coordinates for initial resize
        if bboxes is not None:
            bboxes = bboxes.clone()
            bboxes[:, [0,2]] *= w_scale
            bboxes[:, [1,3]] *= h_scale
            
        if kps is not None:
            kps = kps.clone()
            kps[:, 0::2] *= w_scale
            kps[:, 1::2] *= h_scale
        
        # Calculate valid scale range based on annotations
        if bboxes is not None:
            bbox = bboxes[0]  # Assuming single bbox
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            min_scale, max_scale = self.calculate_valid_scale_range(
                bbox_width, bbox_height, new_w, new_h)
        else:
            min_scale, max_scale = self.scale
            
        # Random scale from valid range
        scale = random.uniform(min_scale, max_scale)
        
        if scale > 1.0:  # Zoom in
            # Calculate crop size
            crop_size = int(self.input_size / scale)
            
            # Calculate valid crop range
            valid_left, valid_right, valid_top, valid_bottom = self.calculate_valid_crop_range(
                bboxes[0] if bboxes is not None else None,
                kps,
                crop_size,
                new_w,
                new_h
            )
            
            # Random crop position within valid range
            crop_x = int(random.uniform(valid_left, valid_right)) if valid_right > valid_left else valid_left
            crop_y = int(random.uniform(valid_top, valid_bottom)) if valid_bottom > valid_top else valid_top
            
            # Apply crop
            image = TF.crop(image, crop_y, crop_x, crop_size, crop_size)
            
            # Update coordinates for crop
            if bboxes is not None:
                bboxes[:, [0,2]] -= crop_x
                bboxes[:, [1,3]] -= crop_y
                
            if kps is not None:
                kps[:, 0::2] -= crop_x
                kps[:, 1::2] -= crop_y
                
            # Final resize to desired input size
            image = TF.resize(image, (self.input_size, self.input_size), self.interpolation)
            
            # Final scale adjustment for coordinates
            final_scale = self.input_size / crop_size
            if bboxes is not None:
                bboxes *= final_scale
            if kps is not None:
                kps *= final_scale
                
        else:  # Zoom out or no zoom
            pad_w = max(0, self.input_size - new_w)
            pad_h = max(0, self.input_size - new_h)
            padding = [pad_w//2, pad_h//2, pad_w-(pad_w//2), pad_h-(pad_h//2)]
            
            # Apply padding
            image = TF.pad(image, padding, fill=0)
            if image.size[0] != 224 or image.size[1] != 224:
                print(image.size)
                exit(1)
            
            # Update coordinates for padding
            if bboxes is not None:
                bboxes[:, [0,2]] += pad_w//2
                bboxes[:, [1,3]] += pad_h//2
                
            if kps is not None:
                kps[:, 0::2] += pad_w//2
                kps[:, 1::2] += pad_h//2
                
        return image, bboxes, kps

class OCRSafeAugment:
    def __init__(self, strength=0.5):
        """
        Initialize OCR-safe augmentation with controllable strength.
        
        Args:
            strength (float): Global strength of augmentations, from 0.0 to 1.0
        """
        self.strength = strength
        
    def apply_perspective(self, img):
        """Safe perspective transform that preserves text readability using torchvision"""
        width, height = img.size
        
        # Calculate safe perspective points that won't cut off text
        margin_w = int(width * 0.1 * self.strength)
        margin_h = int(height * 0.1 * self.strength)
        
        # Define start points (original image corners)
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        
        # Define end points (randomly perturbed corners within safe margins)
        endpoints = [
            [random.randint(0, margin_w), random.randint(0, margin_h)],  # top-left
            [width - 1 - random.randint(0, margin_w), random.randint(0, margin_h)],  # top-right
            [width - 1 - random.randint(0, margin_w), height - 1 - random.randint(0, margin_h)],  # bottom-right
            [random.randint(0, margin_w), height - 1 - random.randint(0, margin_h)]  # bottom-left
        ]
        
        # Convert image to tensor if it's not already
        if not isinstance(img, torch.Tensor):
            img_tensor = transforms.ToTensor()(img)
        else:
            img_tensor = img
            
        # Apply perspective transform
        transformed_img = TF.perspective(
            img_tensor,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=[0, 0, 0]  # black fill for areas outside the transform
        )
        
        # Convert back to PIL if input was PIL
        if isinstance(img, Image.Image):
            transformed_img = transforms.ToPILImage()(transformed_img)
            
        return transformed_img
    
    def apply_elastic_transform(self, img):
        """Elastic deformation that maintains character integrity"""
        img_tensor = transforms.ToTensor()(img)
        _, h, w = img_tensor.shape
        
        # Generate displacement fields
        grid_scale = 4  # Larger value = more subtle distortion
        dx = torch.rand(h // grid_scale, w // grid_scale) * 2 - 1
        dy = torch.rand(h // grid_scale, w // grid_scale) * 2 - 1
        
        # Upscale displacement fields and apply smoothing
        dx = F.interpolate(dx.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic')[0, 0]
        dy = F.interpolate(dy.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic')[0, 0]
        
        # Scale displacement based on strength
        elastic_strength = min(2.0, self.strength)
        displacement_scale = 0.01 * elastic_strength
        dx *= displacement_scale * w
        dy *= displacement_scale * h
        
        # Create sampling grid
        x_grid = torch.arange(w).float().repeat(h, 1)
        y_grid = torch.arange(h).float().repeat(w, 1).t()
        
        x_grid = x_grid + dx
        y_grid = y_grid + dy
        
        # Normalize coordinates to [-1, 1]
        x_grid = 2 * (x_grid / (w - 1)) - 1
        y_grid = 2 * (y_grid / (h - 1)) - 1
        
        # Stack and reshape
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
        
        # Apply sampling grid
        img_tensor = F.grid_sample(img_tensor.unsqueeze(0), grid, align_corners=True)[0]
        
        return transforms.ToPILImage()(img_tensor)
    
    def apply_color_jitter(self, img):
        """Apply color jittering with controlled intensity"""
        factors = {
            'brightness': random.uniform(1 - 0.4 * self.strength, 1 + 0.4 * self.strength),
            'contrast': random.uniform(1 - 0.4 * self.strength, 1 + 0.4 * self.strength),
            'saturation': random.uniform(1 - 0.4 * self.strength, 1 + 0.4 * self.strength),
            'hue': random.uniform(-0.2 * self.strength, 0.2 * self.strength)
        }
        
        for factor, value in factors.items():
            if factor == 'brightness':
                img = ImageEnhance.Brightness(img).enhance(value)
            elif factor == 'contrast':
                img = ImageEnhance.Contrast(img).enhance(value)
            elif factor == 'saturation':
                img = ImageEnhance.Color(img).enhance(value)
            elif factor == 'hue':
                img = transforms.functional.adjust_hue(img, value)
        return img
    
    def apply_blur(self, img):
        """Apply slight blur with controlled intensity"""
        radius = self.strength * 0.5  # Max blur radius of 0.5 pixels
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    def __call__(self, img):
        """Apply all augmentations with random chance"""
        augmentations = [
            (self.apply_perspective, 0.5),
            (self.apply_elastic_transform, 0.5),
            (self.apply_color_jitter, 0.7),
            (self.apply_blur, 0.3)
        ]
        
        for aug_func, prob in augmentations:
            if random.random() < prob:
                img = aug_func(img)
        
        return img

# Create the complete transformation pipeline
def create_ocr_transform(augment_strength=1.0):
    return transforms.Compose([
        OCRSafeAugment(strength=augment_strength),
        transforms.ToTensor(),
        # Add any additional transforms here
    ])



class SyntheticOCRDataset(Dataset):
    def __init__(self, vocab, seq_length=10, num_samples=1000, img_height=32, img_width=128):
        self.vocab = vocab
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.data = self._generate_data()
        
        # Define augmentations
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def _generate_data(self):
        data = []
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}. Please install it or provide a different font path.")
        
        for _ in range(self.num_samples):
            text = ''.join(random.choices(self.vocab[1:], k=self.seq_length))
            
            # Create RGB image with random background color
            background_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            image = Image.new('RGB', (self.img_width, self.img_height), color=background_color)
            draw = ImageDraw.Draw(image)
            
            # Randomly adjust font size
            font_size = random.randint(int(self.img_height * 0.65), int(self.img_height * 0.85))
            font = ImageFont.truetype(font_path, font_size)
            
            # Calculate text size using font.getbbox instead of draw.textsize
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Randomly position text while ensuring it fits within the image
            max_x = max(0, self.img_width - text_width)
            max_y = max(0, self.img_height - text_height)
            position = (random.randint(0, int(max_x*.5)), random.randint(0, int(max_y*.5)))
            
            # Choose a random dark color for text
            text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            
            # Rotate text
            rotation = random.uniform(-5, 5)
            draw.text(position, text, font=font, fill=text_color)
            image = image.rotate(rotation, expand=True, fillcolor=background_color)
            draw = ImageDraw.Draw(image)
            
            # Crop image back to original size
            left = (image.width - self.img_width) / 2
            top = (image.height - self.img_height) / 2
            right = (image.width + self.img_width) / 2
            bottom = (image.height + self.img_height) / 2
            image = image.crop((left, top, right, bottom))
            
            image_np = np.array(image).astype(np.float32) / 255.0
            data.append((image_np, text))

        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, text = self.data[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = self.transform(image)
        text_indices = [self.vocab.index(char) for char in text]
        return image, torch.tensor(text_indices)
