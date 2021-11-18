# Rethinking Re-Sampling in Imbalanced Semi-Supervised Learning

## Dependencies

* `python3`
* `pytorch`
* `torchvision`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

### Command for reproducing results in the paper 
To train a model on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, random sampler for labeled data and random sampler for unlabeled data
```
python3 fix_train.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler random --semi-sampler random --out cifar10_fix_100_2_random_random
```

To fine-tune a model (here the model trained with above command) on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, mean sampler for labeled data and mean sampler for unlabeled data
```
python3 fix_finetune.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler mean --semi-sampler mean --resume cifar10_fix_100_2_random_random/checkpoint.pth.tar --out cifar10_fix_100_2_random_random_stage2
```

To train a Bi-Sampling model on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2, random sampler + random sampler for the first stage and mean sampler + mean sampler for the second stage
```
python3 fix_BiS.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--sampler1 random --semi-sampler1 random --sampler2 mean --semi-sampler2 mean --out cifar10_fix_100_2_BiS
```

To analyze the per-class precision and recall of a pertained model on CIFAR-10 with imbalanced ratio $\beta$ = 100,  unlabeled ratio $\lambda$ = 2

```
python3 fix_analysis.py --gpu 0 --dataset cifar10 --imb_ratio 100 --ratio 2 \
--resume cifar10_fix_100_2_BiS/checkpoint.pth.tar
```

