import copy
import logging

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client

logger = logging.getLogger(__name__)


class LLMMultiLoRAClient(Client):
    """
    Client implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(LLMMultiLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

    def callback_funcs_for_model_para(self, message: Message):
        # Here we just activate self local LoRA
        self.model.set_active_adapter(f'Client_{self.ID}')
        # logger.info(self.trainer.ctx.model.get_active_adapter())

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        self.state = round
        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            # for these cases (1) unseen client (2) isolated_global_mode,
            # we do not local train and upload local model
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(f"[Local/Global mode] Client #{self.ID} has been "
                            f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()
            sample_size, model_para_all, results = self.trainer.train()
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
                model_para_all = {
                    key: value
                    for key, value in model_para_all.items()
                    if f'Client_{self.ID}.' in key
                }
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                     save_file_name="")

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                  instance_number=sample_size),
                    content=(sample_size, model_para_all)))
