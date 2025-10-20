from mmengine.hooks import Hook

class FreezeExceptBackboneHook(Hook):
    def before_train(self, runner):
        model = runner.model
        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            if name.startswith('img_backbone.'):
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
                frozen += p.numel()
        runner.logger.info(f'FreezeExceptBackboneHook: frozen={frozen}, trainable={trainable}')