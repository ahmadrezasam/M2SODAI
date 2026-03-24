from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeExceptProjectorHook(Hook):
    """
    Custom hook to freeze the entire model except the `channel_projector` layer.
    """
    def before_train_epoch(self, runner):
        model = runner.model
        
        # 1. Freeze parameters
        for name, param in model.named_parameters():
            if 'channel_projector' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # 2. Freeze BN stats and dropout
        for name, module in model.named_modules():
            # Skip the root module to prevent recursion logic issues elsewhere, 
            # though not strictly necessary since we just set eval()
            if name != '' and 'channel_projector' not in name:
                if hasattr(module, 'eval'):
                    module.eval()
                    
        runner.logger.info("[\u2713] FreezeExceptProjectorHook applied: Backbone frozen.")
