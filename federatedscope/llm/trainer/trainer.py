import torch
import logging

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None

import accelerate
from accelerate import Accelerator

from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel

logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        self.grad_accum_step = max(config.llm.grad_accum_step,
                                   config.grad.grad_accum_count)

        if config.llm.accelerator.use:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.grad_accum_step)
            device = self.accelerator.device

        super().__init__(model, data, device, config, only_for_eval, monitor)

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set, run_step=-1):
        if run_step == -1:
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")
        for batch_i in range(run_step):
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            if hasattr(self, 'accelerator'):
                # Build gradient accumulation upon accelerator
                with self.accelerator.accumulate(self.ctx.model):
                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)

            else:
                for idx in range(self.grad_accum_step):
                    self.ctx.cur_batch_i = CtxVar(
                        batch_i * self.grad_accum_step + idx, LIFECYCLE.BATCH)

                    for hook in hooks_set["on_batch_start"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_backward"]:
                        hook(self.ctx)

                    for hook in hooks_set["on_batch_end"]:
                        hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [
                    MODE.TRAIN, MODE.FINETUNE
            ] and self.ctx.cur_epoch_i == self.ctx.num_train_epoch - 1:
                if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                    break

    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use and \
               not ctx.cfg.llm.accelerator.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            # prepare model sharding
            ctx.model.sharding()

            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.optimizer.zero_grad()
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

            ctx.model, ctx.optimizer, ctx.train_loader, \
                ctx.val_loader, ctx.test_loader, ctx.scheduler = \
                self.accelerator.prepare(
                    ctx.model,
                    getattr(ctx, 'optimizer', None),
                    getattr(ctx, 'train_loader', None),
                    getattr(ctx, 'val_loader', None),
                    getattr(ctx, 'test_loader', None),
                    getattr(ctx, 'scheduler', None)
                )

        elif ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()

        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids']
            labels = ctx.data_batch['labels']
            attention_mask = ctx.data_batch['attention_mask']
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(ctx.loss_task)
            ctx.optimizer.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()
            ctx.optimizer.zero_grad()

        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()

            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                   ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        setattr(ctx, 'eval_metrics', eval_results)

        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
