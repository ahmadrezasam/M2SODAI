# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early Stopping Hook.

    Args:
        monitor (str): The metric to monitor. Default: 'bbox_mAP'.
        patience (int): Number of epochs with no improvement after which training will be stopped.
            Default: 5.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            Default: 0.0.
        rule (str): Comparison rule. Options are 'less' and 'greater'. Default: 'greater'.
    """

    def __init__(self,
                 monitor='bbox_mAP',
                 patience=5,
                 min_delta=0.0,
                 rule='greater'):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.rule = rule
        self.best_score = None
        self.wait_count = 0

        assert rule in ['less', 'greater']
        if rule == 'less':
            self.compare_func = lambda a, b: a < (b - min_delta)
        else:
            self.compare_func = lambda a, b: a > (b + min_delta)

    def after_train_epoch(self, runner):
        # We only check at the end of each evaluation
        # In MMDetection, evaluation results are stored in runner.log_buffer.output
        if not runner.log_buffer.output:
            return

        if self.monitor not in runner.log_buffer.output:
            # If the metric isn't in the logs yet (e.g., no evaluation this epoch)
            return

        current_score = runner.log_buffer.output[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
            self.wait_count = 0
        elif self.compare_func(current_score, self.best_score):
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            runner.logger.info(
                f"\nEarly stopping triggered! No improvement in {self.monitor} "
                f"for {self.patience} evaluations. Best score: {self.best_score:.4f}")
            # Stop the training
            import sys
            # MMDetection runner doesn't have a clean 'stop' flag in older versions, 
            # so we set max_epochs to current epoch to finish naturally.
            runner._max_epochs = runner.epoch
