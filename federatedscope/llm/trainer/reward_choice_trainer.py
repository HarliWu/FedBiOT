import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken

import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


def cal_loss(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    # new_logits = logits * mask.unsqueeze(-1)
    # new_logits = torch.sum(new_logits, dim=1)[:, choices]
    # new_labels = torch.sum(new_labels, dim=1)

    new_logits = shift_logits[..., choices]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1))
    # return new_logits.view(-1, len(choices)), new_labels.view(-1), loss
    return new_logits, new_labels, loss


class RewardChoiceTrainer(LLMTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        try:
            self.choices = []
            for choice in config.trainer.choices:
                self.choices.append(
                    self.tokenizer(f': {choice}')['input_ids'][-1])
                # assert len(self.choices[-1]) == 1
            logger.info(f'Choice indices: {self.choices}')
        except AssertionError:
            raise AssertionError('The choice should be limited to one token.')
        except:
            raise ValueError('trainer.choices not found in the config.')

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)

        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)

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
        new_logits, new_labels, loss = cal_loss(logits, labels, self.choices)

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # logger.info(f'{input_ids}')
        # logger.info(f'{labels}')

        new_labels = new_labels.view(-1)
        new_logits = new_logits.view(-1, len(self.choices))
        new_logits = new_logits[(
            new_labels != DefaultToken.IGNORE_INDEX.value), :]
        new_labels = new_labels[(new_labels !=
                                 DefaultToken.IGNORE_INDEX.value)]
        _, predicted = new_logits.max(1)
        # logger.info(f'{predicted}, {new_labels}, {new_logits}')

        ctx.y_true = CtxVar(new_labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(new_logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true)
        ctx.ys_pred.append(ctx.y_pred)

    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(torch.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(torch.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)


def call_reward_choice_trainer(trainer_type):
    if trainer_type == 'llmrewardchoicetrainer':
        trainer_builder = RewardChoiceTrainer
        return trainer_builder


register_trainer('llmrewardchoicetrainer', call_reward_choice_trainer)
