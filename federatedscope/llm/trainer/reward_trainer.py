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


def _get_batch_logps(logits, labels, average_log_prob=False):
    """
    Source: https://github.com/eric-mitchell/direct-preference-optimization/
        blob/main/trainers.py#L208

    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities.
            Label tokens with a value of -100 are ignored.
            Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability
            per (non-masked) token. Otherwise, return the sum of the
            log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum
            log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != DefaultToken.IGNORE_INDEX.value)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == DefaultToken.IGNORE_INDEX.value] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1),
                                   dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(policy_chosen_logps,
             policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps,
             beta,
             reference_free=False):
    """
    Source: https://github.com/eric-mitchell/direct-preference-optimization/
        blob/main/trainers.py#L208

    Compute the DPO loss for a batch of policy and reference
    model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model
            for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model
            for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model
            for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model
            for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something
            in the range of 0.1 to 0.5. We ignore the reference model
            as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model
            and implicitly use a reference model that assigns equal probability
            to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards
            for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps -
                               reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


class DPORewardTrainer(LLMTrainer):
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

            with torch.no_grad():
                ref_win_logps, ref_lose_logps = \
                    self._batch_forward(ctx, win_input_ids, win_labels,
                                        win_attention_mask, lose_input_ids,
                                        lose_labels, lose_attention_mask,
                                        disable_adapter=True)

            adap_win_logps, adap_lose_logps = \
                self._batch_forward(ctx, win_input_ids, win_labels,
                                    win_attention_mask, lose_input_ids,
                                    lose_labels, lose_attention_mask,
                                    disable_adapter=False)

        elif ctx.cfg.llm.deepspeed.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            with torch.no_grad():
                ref_win_logps, ref_lose_logps = \
                    self._batch_forward_deepspeed(
                        ctx, win_input_ids, win_labels, win_attention_mask,
                        lose_input_ids, lose_labels, lose_attention_mask,
                        disable_adapter=True)

            adap_win_logps, adap_lose_logps = \
                self._batch_forward_deepspeed(
                    ctx, win_input_ids, win_labels, win_attention_mask,
                    lose_input_ids, lose_labels, lose_attention_mask,
                    disable_adapter=False)

        else:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            with torch.no_grad():
                ref_win_logps, ref_lose_logps = \
                    self._batch_forward(ctx, win_input_ids, win_labels,
                                        win_attention_mask, lose_input_ids,
                                        lose_labels, lose_attention_mask,
                                        disable_adapter=True)

            adap_win_logps, adap_lose_logps = \
                self._batch_forward(ctx, win_input_ids, win_labels,
                                    win_attention_mask, lose_input_ids,
                                    lose_labels, lose_attention_mask,
                                    disable_adapter=False)

        # loss follows using Equation (7) of Direct Preference Optimization:
        # Your Language Model is Secretly a Reward Model
        loss, win_rewards, lose_rewards = dpo_loss(adap_win_logps,
                                                   adap_lose_logps,
                                                   ref_win_logps,
                                                   ref_lose_logps,
                                                   beta=self.reward_coeff)

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(torch.zeros(len(win_input_ids)), LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(
            torch.where(win_rewards.cpu() > lose_rewards.cpu(),
                        torch.zeros(len(win_input_ids)),
                        torch.ones(len(win_input_ids))), LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(win_input_ids), LIFECYCLE.BATCH)

    def _batch_forward(self,
                       ctx,
                       win_input_ids,
                       win_labels,
                       win_attention_mask,
                       lose_input_ids,
                       lose_labels,
                       lose_attention_mask,
                       disable_adapter=False):
        win_outputs = ctx.model(disable_adapter=disable_adapter,
                                input_ids=win_input_ids,
                                labels=win_labels,
                                attention_mask=win_attention_mask)
        win_logps = _get_batch_logps(win_outputs.logits,
                                     win_labels,
                                     average_log_prob=False)

        lose_outputs = ctx.model(disable_adapter=disable_adapter,
                                 input_ids=lose_input_ids,
                                 labels=lose_labels,
                                 attention_mask=lose_attention_mask)
        lose_logps = _get_batch_logps(lose_outputs.logits,
                                      lose_labels,
                                      average_log_prob=False)

        return win_logps, lose_logps

    def _batch_forward_deepspeed(self,
                                 ctx,
                                 win_input_ids,
                                 win_labels,
                                 win_attention_mask,
                                 lose_input_ids,
                                 lose_labels,
                                 lose_attention_mask,
                                 disable_adapter=False):
        win_outputs = ctx.model_engine(disable_adapter=disable_adapter,
                                       input_ids=win_input_ids,
                                       labels=win_labels,
                                       attention_mask=win_attention_mask)
        win_logps = _get_batch_logps(win_outputs.logits,
                                     win_labels,
                                     average_log_prob=False)

        lose_outputs = ctx.model_engine(disable_adapter=disable_adapter,
                                        input_ids=lose_input_ids,
                                        labels=lose_labels,
                                        attention_mask=lose_attention_mask)
        lose_logps = _get_batch_logps(lose_outputs.logits,
                                      lose_labels,
                                      average_log_prob=False)

        return win_logps, lose_logps

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
    if trainer_type == 'llmdporewardtrainer':
        trainer_builder = DPORewardTrainer
        return trainer_builder


register_trainer('llmdporewardtrainer', call_reward_trainer)
