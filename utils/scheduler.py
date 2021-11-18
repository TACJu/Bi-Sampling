import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_epochs, e_total, cycles=.5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.e_total = e_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return max(1e-5, float(epoch) / float(max(1.0, self.warmup_epochs)))
        # progress after warmup
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.e_total - self.warmup_epochs))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))