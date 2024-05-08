import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken

import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


class PPORewardTrainer(LLMTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.reward_coeff = config.llm.reward_coeff

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)

        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            win_outputs = ctx.model(input_ids=win_input_ids,
                                    attention_mask=win_attention_mask)
            lose_outputs = ctx.model(input_ids=lose_input_ids,
                                     attention_mask=lose_attention_mask)

            win_rewards = win_outputs.logits
            lose_rewards = lose_outputs.logits

        elif ctx.cfg.llm.deepspeed.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            win_outputs = ctx.model_engine(input_ids=win_input_ids,
                                           attention_mask=win_attention_mask)
            lose_outputs = ctx.model_engine(input_ids=lose_input_ids,
                                            attention_mask=lose_attention_mask)

            win_rewards = win_outputs.logits
            lose_rewards = lose_outputs.logits

        else:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            win_outputs = ctx.model(input_ids=win_input_ids,
                                    attention_mask=win_attention_mask)
            lose_outputs = ctx.model(input_ids=lose_input_ids,
                                     attention_mask=lose_attention_mask)

            win_rewards = win_outputs.logits
            lose_rewards = lose_outputs.logits

        # loss of reward model training, following the paper
        # Learning to summarize from human feedback
        loss = -F.logsigmoid(win_rewards - lose_rewards).mean()

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        win_rewards = win_rewards.view(-1)
        lose_rewards = lose_rewards.view(-1)
        ctx.y_true = CtxVar(torch.zeros(len(win_input_ids)), LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(
            torch.where(win_rewards.cpu() > lose_rewards.cpu(),
                        torch.zeros(len(win_input_ids)),
                        torch.ones(len(win_input_ids))), LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(win_input_ids), LIFECYCLE.BATCH)

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

        # move the training data to cpu
        ctx.data_batch['win_input_ids'].cpu()
        ctx.data_batch['win_labels'].cpu()
        ctx.data_batch['win_attention_mask'].cpu()
        ctx.data_batch['lose_input_ids'].cpu()
        ctx.data_batch['lose_labels'].cpu()
        ctx.data_batch['lose_attention_mask'].cpu()

    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.append(ctx.y_pred.detach().cpu().numpy())

    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)


def call_reward_trainer(trainer_type):
    if trainer_type == 'llmpporewardtrainer':
        trainer_builder = PPORewardTrainer
        return trainer_builder


register_trainer('llmpporewardtrainer', call_reward_trainer)
