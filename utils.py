# Tons of helper functions, some mine, most from https://github.com/open-mmlab/mmdetection

from typing import Optional, Tuple, Union, Any
from functools import partial
import os

import wandb
import numpy as np
import torch
from torch import Tensor
import torchvision
import Levenshtein as lev
from matplotlib import pyplot as plt

array_like_type = Union[Tensor, np.ndarray]

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


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list


# This function is modified from: https://github.com/pytorch/vision/
class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, bboxes: Tensor, scores: Tensor, iou_threshold: float,
                offset: int, score_threshold: float, max_num: int) -> Tensor:
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # inds = nms(
        #     bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)

        inds = torchvision.ops.nms(bboxes, scores, iou_threshold)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds


def nms(boxes: array_like_type,
        scores: array_like_type,
        iou_threshold: float,
        offset: int = 0,
        score_threshold: float = 0,
        max_num: int = -1) -> Tuple[array_like_type, array_like_type]:
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have
        the same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, (Tensor, np.ndarray))
    assert isinstance(scores, (Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = NMSop.apply(boxes, scores, iou_threshold, offset, score_threshold,
                       max_num)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds

def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: Optional[dict],
                class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop('type', 'nms')
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, nms_cfg_['iou_threshold'])#, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep

def scale_boxes(boxes,
                scale_factor: Tuple[float, float]):
    """Scale boxes with type of tensor or box type. """
    repeat_num = int(boxes.size(-1) / 2)
    scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
    return boxes * scale_factor

def get_box_wh(boxes: Tensor) -> Tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or box type.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes with type of tensor
            or box type.

    Returns:
        Tuple[Tensor, Tensor]: the width and height of boxes.
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w, h


def anchor_inside_flags(flat_anchors: Tensor,
                        valid_flags: Tensor,
                        img_shape: Tuple[int],
                        allowed_border: int = 0) -> Tensor:
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape=None) -> Tensor:
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
            optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            # TODO: delete
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes

def kps2distance(points, kps, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        kps (Tensor): Shape (n, K), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """

    preds = []
    for i in range(0, kps.shape[1], 2):
        px = kps[:, i] - points[:, i%2]
        py = kps[:, i+1] - points[:, i%2+1]
        if max_dis is not None:
            px = px.clamp(min=0, max=max_dis - eps)
            py = py.clamp(min=0, max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, axis=-1)

def bbox2distance(points: Tensor,
                  bbox: Tensor,
                  max_dis: float = None,
                  eps: float = 0.1) -> Tensor:
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2) or (b, n, 2), [x, y].
        bbox (Tensor): Shape (n, 4) or (b, n, 4), "xyxy" format
        max_dis (float, optional): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    """

    def fp16_clamp(x, min=None, max=None):
        if not x.is_cuda and x.dtype == torch.float16:
            # clamp for cpu float16, tensor fp16 has no clamp implementation
            return x.float().clamp(min, max).half()

        return x.clamp(min, max)

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def bbox_center_distance(bboxes: Tensor, priors: Tensor) -> Tensor:
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        priors (Tensor): Shape (n, 4) for priors, "xyxy" format.

    Returns:
        Tensor: Center distances between bboxes and priors.
    """
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = torch.stack((bbox_cx, bbox_cy), dim=1)

    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    priors_points = torch.stack((priors_cx, priors_cy), dim=1)

    distances = (priors_points[:, None, :] -
                 bbox_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances

def iou_calculator(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate IoU between 2D bboxes.

    Args:
        bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
            in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
            y2, score> format.
        bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
            in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
            score> format, or be empty. If ``is_aligned `` is ``True``,
            then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection
            over foreground), or "giou" (generalized intersection over
            union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    # bboxes1 = get_box_tensor(bboxes1)
    # bboxes2 = get_box_tensor(bboxes2)
    assert bboxes1.size(-1) in [0, 4, 5]
    assert bboxes2.size(-1) in [0, 4, 5]
    if bboxes2.size(-1) == 5:
        bboxes2 = bboxes2[..., :4]
    if bboxes1.size(-1) == 5:
        bboxes1 = bboxes1[..., :4]

    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

def assign(anchors, num_level_priors, gt_bboxes, gt_bboxes_ignore, gt_labels, topk=9):
    """Assign gt to priors.

    The assignment is done in following steps

    1. compute iou between all prior (prior of all pyramid levels) and gt
    2. compute center distance between all prior and gt
    3. on each pyramid level, for each gt, select k prior whose center
        are closest to the gt center, so we total select k*l prior as
        candidates for each gt
    4. get corresponding iou for the these candidates, and compute the
        mean and std, set mean + std as the iou threshold
    5. select these candidates whose iou are greater than or equal to
        the threshold as positive
    6. limit the positive sample's center in gt

    If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
    are not None, the overlaps calculation in the first step
    will also include dynamic cost, which is currently only used in
    the DDOD.

    Args:

    Returns:
        dict: The assign result.
    """
    priors = anchors

    INF = 100000000
    priors = priors[:, :4]
    num_gt, num_priors = gt_bboxes.size(0), priors.size(0)

    # compute iou between all bbox and gt
    overlaps = iou_calculator(priors, gt_bboxes)

    # assign 0 by default
    assigned_gt_inds = overlaps.new_full(
        (num_priors, ), 0, dtype=torch.long)

    if num_gt == 0 or num_priors == 0:
        # No ground truth or boxes, return empty assignment
        max_overlaps = overlaps.new_zeros((num_priors, ))
        if num_gt == 0:
            # No truth, assign everything to background
            assigned_gt_inds[:] = 0
        assigned_labels = overlaps.new_full((num_priors, ),
                                            -1,
                                            dtype=torch.long)
        return dict(
            num_gt=num_gt,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)


    # compute center distance between all bbox and gt
    distances = bbox_center_distance(gt_bboxes, priors)

    # Selecting candidates based on the center distance
    candidate_idxs = []
    start_idx = 0
    for level, priors_per_level in enumerate(num_level_priors):
        # on each pyramid level, for each gt,
        # select k bbox whose center are closest to the gt center
        end_idx = start_idx + priors_per_level
        distances_per_level = distances[start_idx:end_idx, :]
        selectable_k = min(topk, priors_per_level)
        _, topk_idxs_per_level = distances_per_level.topk(
            selectable_k, dim=0, largest=False)
        candidate_idxs.append(topk_idxs_per_level + start_idx)
        start_idx = end_idx
    candidate_idxs = torch.cat(candidate_idxs, dim=0)

    # get corresponding iou for the these candidates, and compute the
    # mean and std, set mean + std as the iou threshold
    candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
    overlaps_mean_per_gt = candidate_overlaps.mean(0)
    overlaps_std_per_gt = candidate_overlaps.std(0)
    overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

    is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

    # limit the positive sample's center in gt
    for gt_idx in range(num_gt):
        candidate_idxs[:, gt_idx] += gt_idx * num_priors
    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    ep_priors_cx = priors_cx.view(1, -1).expand(
        num_gt, num_priors).contiguous().view(-1)
    ep_priors_cy = priors_cy.view(1, -1).expand(
        num_gt, num_priors).contiguous().view(-1)
    candidate_idxs = candidate_idxs.view(-1)

    # calculate the left, top, right, bottom distance between positive
    # prior center and gt side
    l_ = ep_priors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
    t_ = ep_priors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
    r_ = gt_bboxes[:, 2] - ep_priors_cx[candidate_idxs].view(-1, num_gt)
    b_ = gt_bboxes[:, 3] - ep_priors_cy[candidate_idxs].view(-1, num_gt)
    is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

    is_pos = is_pos & is_in_gts

    # if an anchor box is assigned to multiple gts,
    # the one with the highest IoU will be selected.
    overlaps_inf = torch.full_like(overlaps,
                                    -INF).t().contiguous().view(-1)
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
    overlaps_inf = overlaps_inf.view(num_gt, -1).t()

    max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
    assigned_gt_inds[
        max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

    assigned_labels = assigned_gt_inds.new_full((num_priors, ), -1)
    pos_inds = torch.nonzero(
        assigned_gt_inds > 0, as_tuple=False).squeeze()
    if pos_inds.numel() > 0:
        assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                1]
    return dict(
        num_gt=num_gt,
        gt_inds=assigned_gt_inds,
        max_overlaps=max_overlaps,
        labels=assigned_labels)

