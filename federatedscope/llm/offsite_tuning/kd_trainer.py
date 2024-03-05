import torch
import logging
import copy

from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

logger = logging.getLogger(__name__)


def l2_norm(output_student_float, output_teacher_float):
    std = output_teacher_float.pow(2).mean().sqrt()
    return (output_teacher_float - output_student_float).div(std).pow(2).mean()


def get_kd_loss(loss_fn, raw_model, adap_model, layerwise_distill=False):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
    layerwise_distill = (layerwise_distill
                         and hasattr(adap_model, 'teacher_model_mapping'))
    kwargs = adap_model.student_l.input_kwargs
    args = adap_model.student_l.input_args
    output_teacher = args[0]
    output_student = copy.deepcopy(args[0])
    args = list(args[1:])
    args = tuple(args)

    kd_loss = 0.0
    with torch.no_grad():
        raw_model.teacher.eval()

        if layerwise_distill:
            student_teacher_map = adap_model.teacher_model_mapping
            teacher_outputs = [0] * len(student_teacher_map)

        for i, teacher_layer in enumerate(raw_model.teacher):
            output_teacher = teacher_layer(output_teacher, *args, **kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]
            if layerwise_distill and (i in student_teacher_map):
                # map with the teacher's model and accumulate kd_loss
                teacher_outputs[student_teacher_map.index(
                    i)] = output_teacher.float()

    if layerwise_distill:
        adap_model_training_state = adap_model.student.training
        adap_model.student.eval()

        for layer, output_teacher_float in zip(adap_model.student,
                                               teacher_outputs):
            output_student = layer(output_student, *args, **kwargs)
            if isinstance(output_student, tuple):
                output_student = output_student[0]
            output_student_float = output_student.float()
            kd_loss += loss_fn(output_student_float, output_teacher_float)

        adap_model.student.train(mode=adap_model_training_state)
    else:
        output_student_float = adap_model.student_r.cached_output.float()
        output_teacher_float = output_teacher.float()
        kd_loss = loss_fn(output_student_float, output_teacher_float)

    return kd_loss


class KDTrainer(LLMTrainer):
    def __init__(self,
                 raw_model,
                 adapter_model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(KDTrainer, self).__init__(adapter_model, data, device, config,
                                        only_for_eval, monitor)
        self.ctx.raw_model = raw_model.to(device)
        self.lm_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.lm_loss_weight
        self.kd_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.kd_loss_weight

    def _hook_on_fit_start_numerical_precision(self, ctx):
        super(KDTrainer, self)._hook_on_fit_start_numerical_precision(ctx)
        # if self.cfg.train.is_enable_half:
        #     ctx.raw_model.to(torch.bfloat16)

    def train(self, target_data_split_name="train", hooks_set=None):
        num_samples, model_para_all, eval_metrics = \
            super(KDTrainer, self).train(target_data_split_name, hooks_set)
        logger.info("Finish alignment, move raw model to cpu.")
        self.ctx.raw_model.cpu()
        return num_samples, model_para_all, eval_metrics

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask)

        logits = outputs.logits
        kd_loss = self.kd_loss_weight * get_kd_loss(l2_norm, ctx.raw_model,
                                                    ctx.model)
        lm_loss = self.lm_loss_weight * outputs.loss
        loss = kd_loss + lm_loss

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

        logger.info(f'lm_loss: {lm_loss.item()}, kd loss: {kd_loss.item()}')
