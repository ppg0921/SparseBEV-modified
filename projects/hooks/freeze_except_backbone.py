from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS

@HOOKS.register_module()
class FreezeExceptBackboneHook(Hook):
    """Freeze all params except those under 'img_backbone.' (MMCV Runner)."""
    def before_train_epoch(self, runner):
        model = runner.model
        # unwrap DataParallel/DistributedDataParallel if needed
        if hasattr(model, 'module'):
            model = model.module

        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            if name.startswith('img_backbone.'):
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
                frozen += p.numel()

        runner.logger.info(
            f"[FreezeExceptBackboneHook] trainable={trainable:,} frozen={frozen:,}"
        )