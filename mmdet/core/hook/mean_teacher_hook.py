# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MeanTeacherEpochHook(Hook):
    """Hook that updates the Mean Teacher detector's current epoch.

    This hook calls ``model.set_epoch(epoch)`` at the beginning of each epoch
    so the detector knows when to enable pseudo-label losses (after burn-in)
    and how to decay the pseudo-label confidence threshold.
    """

    def before_train_epoch(self, runner):
        model = runner.model
        # Handle DataParallel/DistributedDataParallel wrappers
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'set_epoch'):
            model.set_epoch(runner.epoch)
            runner.logger.info(
                f'[MeanTeacherEpochHook] Set epoch={runner.epoch}')
