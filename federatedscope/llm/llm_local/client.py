import copy
import logging
import torch
import random

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.data import ClientData

logger = logging.getLogger(__name__)


def reorg_client_data(cdata: ClientData):
    from torch.utils.data import random_split, ConcatDataset
    data_length = len(cdata.train_data)
    val_data_length = max(min(int(data_length * 0.05), 200), 1)
    train_data_length = data_length - val_data_length
    train_data, val_data = random_split(cdata.train_data,
                                        [train_data_length, val_data_length])
    test_data = ConcatDataset([cdata.val_data, cdata.test_data])
    return ClientData(cdata.client_cfg, train_data, val_data, test_data)


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
        if config.llm.adapter.count > 1:
            data = reorg_client_data(data)
        super(LLMMultiLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('adapter_eval',
                               self.callback_funcs_for_adapter_eval,
                               ['grouping'])
        self.register_handlers('set_active_adapter_idx',
                               self.callback_funcs_for_setting_adapter_idx,
                               [None])

    def callback_funcs_for_model_para(self, message: Message):
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

            # Two mode of multilora training: Client-wise and clustering
            if self._cfg.llm.adapter.local_only:
                # This is client-wise
                self.model.set_active_adapter(f'Adapter_{self.ID}')
                adapter_idx = self.ID
            elif self._cfg.llm.adapter.count > 1:
                # This is clustering
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count
                if self._cfg.llm.adapter.warmup.use and \
                        self.state < total_warmup_round:
                    # Initialization for all adapters
                    adapter_idx = self.state // warmup_round
                elif self._cfg.llm.adapter.grouping.use:
                    adapter_idx = self.adapter_idx
                else:
                    # select the adapter with min val loss
                    with torch.no_grad():
                        min_loss, adapter_indices = 10.0, []
                        for i in range(self._cfg.llm.adapter.count):
                            if len(self.data.val_data) == 0:
                                adapter_indices.append(i)
                                continue
                            self.model.set_active_adapter(f'Adapter_{i}')
                            self.model.eval()
                            metrics = self.trainer.evaluate(
                                target_data_split_name='val')
                            logger.info(
                                f'Adapter {i} with the results: {metrics}')
                            if i == 0 or min_loss > metrics['val_avg_loss']:
                                min_loss, adapter_indices = metrics[
                                    'val_avg_loss'], [i]
                            elif min_loss == metrics['val_avg_loss']:
                                adapter_indices.append(i)
                        logger.info(adapter_indices)
                        adapter_idx = random.choice(adapter_indices)
                # activate the selected adapter for further training
                logger.info(
                    f'Activate the adapter {adapter_idx} for training...')
                self.model.set_active_adapter(f'Adapter_{adapter_idx}')
                self.model.train()
            else:
                raise ValueError(
                    'You should set llm.adapter.local_only to True '
                    'or llm.adapter.count > 1')

            sample_size, model_para_all, results = self.trainer.train()
            train_data_size = len(self.data.train_data)
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
                model_para_all = {
                    key: value
                    for key, value in model_para_all.items()
                    if f'Adapter_{adapter_idx}.' in key
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

    def callback_funcs_for_adapter_eval(self, message: Message):
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        metrics = {}
        with torch.no_grad():
            for i in range(self._cfg.llm.adapter.count):
                if len(self.data.val_data) == 0:
                    metrics[f'adapter_{i}_avg_loss'] = random.random()
                    continue
                self.model.set_active_adapter(f'Adapter_{i}')
                self.model.eval()
                adap_metrics = self.trainer.evaluate(
                    target_data_split_name='val')
                logger.info(f'Client {self.ID} Adapter {i} with '
                            f'the results: {adap_metrics}')
                metrics[f'adapter_{i}_avg_loss'] = adap_metrics['val_avg_loss']

        self.comm_manager.send(
            Message(msg_type='grouping',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_setting_adapter_idx(self, message: Message):
        self.adapter_idx = message.content
