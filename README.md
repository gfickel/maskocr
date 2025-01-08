# MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

General purpose state of the art OCR network based on ViT. This is an unofficial implementation, however I tried to be as close as possible to the original.

## Train

I'm currently using [Chinese City Parking Dataset](https://github.com/detectRecog/CCPD) for my training since it is freely available.

## Test

TODO

## Changes from Paper

I've made some changes to make the training faster:

1. Adam from [The Road Less Scheduled](https://arxiv.org/abs/2405.15682), with original implementation from https://github.com/facebookresearch/schedule_free
2. One Cycle Policy from [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120) (implemented [on pytorch](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html))
3. Learning rate finder [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) through [fastai](https://github.com/fastai/fastai)
