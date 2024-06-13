import logging
import torch
import random
import math
from federatedscope.core.message import Message

from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_param_dict

logger = logging.getLogger(__name__)


class LLMMultiLoRAServer(Server):
    """
    Server implementation
    We broadcast the model to each client and ask them to train locally
    Afterward, we collect the model back and save it as checkpoints
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(LLMMultiLoRAServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        if self._cfg.llm.adapter.count > 1:
            self.aggregator.total_train_size = len(data.train_data)
            self.aggregator.num_clients = client_num

        if self._cfg.llm.adapter.local_only:
            logger.warning("In local training mode, we will use all clients. "
                           "And we set the total round to 0 for one training "
                           "round only. ")

            self.sampler = None
            self.sample_client_num = client_num

        if self._cfg.llm.adapter.grouping.use:
            self.msg_buffer['adapter_eval'] = dict()

    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('grouping', self.callback_funcs_for_grouping,
                               ['set_active_adapter_idx'])

    def _start_new_training_round(self, aggregated_num=0, skip_grouping=False):
        if self._cfg.llm.adapter.grouping.use and not skip_grouping:
            total_warmup_round = 0
            if self._cfg.llm.adapter.warmup.use:
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count

            r = self._cfg.llm.adapter.grouping.round
            if self.state >= total_warmup_round and \
                    (self.state - total_warmup_round) % r == 0:
                logger.info('Server: Performing a grouping step...')
                self.broadcast_model_para(msg_type='adapter_eval',
                                          filter_unseen_clients=False)
                return

        super()._start_new_training_round(aggregated_num)

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            # merged_adapter = dict()

            # for client_id in train_msg_buffer.keys():
            #     if self.model_num == 1:
            #         _, model_param = train_msg_buffer[client_id]
            #         # merged_adapter.update(model_param)
            #         for key, value in model_param.items():
            #             if key not in merged_adapter:
            #                 merged_adapter[key] = [value]
            #             else:
            #                 merged_adapter[key].append(value)
            #     else:
            #         train_data_size, model_para_multiple = \
            #             train_msg_buffer[client_id]
            #         # merged_adapter.update(model_para_multiple[model_idx])
            #         for key, value in model_para_multiple[model_idx].items():
            #             if key not in merged_adapter:
            #                 merged_adapter[key] = [value]
            #             else:
            #                 merged_adapter[key].append(value)

            # # calculate the mean
            # for key in merged_adapter.keys():
            #     # logger.info(f'{key}: {len(merged_adapter[key])}')
            #     avg_tensor = torch.zeros_like(merged_adapter[key][0])
            #     for value in merged_adapter[key]:
            #         avg_tensor += (value / len(merged_adapter[key]))
            #     merged_adapter[key] = avg_tensor

            # msg_list = [(1, merged_adapter)]

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }
            # logger.info(f'The staleness is {staleness}')

            warmup_round = self._cfg.llm.adapter.warmup.round
            total_warmup_round = \
                warmup_round * self._cfg.llm.adapter.count
            if self._cfg.llm.adapter.warmup.use and \
                    self.state < total_warmup_round:
                result = aggregator.aggregate(agg_info)
            else:
                result = aggregator.aggregate_on_model(agg_info)

            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return aggregated_num

    def trigger_for_start(self):
        # start feature engineering (This part is for hard code)
        if self.check_client_join_in():
            logger.info('Waited all clients join, start now...')
            self.trigger_for_feat_engr(self.broadcast_model_para, {
                'msg_type': 'adapter_eval',
                'filter_unseen_clients': False,
            })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))
            logger.info('Server: Performing a grouping step...')

    def callback_funcs_for_grouping(self, message: Message):
        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['adapter_eval'].keys():
            self.msg_buffer['adapter_eval'][rnd] = dict()

        self.msg_buffer['adapter_eval'][rnd][sender] = \
            [(i, content[f'adapter_{i}_avg_loss'])
             for i in range(self._cfg.llm.adapter.count)]
        self.msg_buffer['adapter_eval'][rnd][sender] = \
            sorted(self.msg_buffer['adapter_eval'][rnd][sender],
                   key=lambda x: x[1])

        return self.check_and_grouping()

    def check_and_grouping(self):
        if 'adapter_eval' not in self.msg_buffer.keys() or \
                len(self.msg_buffer['adapter_eval'].keys()) == 0:
            return False

        buffer = self.msg_buffer['adapter_eval']
        cur_round = max(buffer.keys())
        cur_buffer = buffer[cur_round]
        if len(cur_buffer) < self.client_num:
            return False

        # convert the list to the iterator
        for sender in cur_buffer.keys():
            cur_buffer[sender] = iter(cur_buffer[sender])

        num_adap = self._cfg.llm.adapter.count
        self.adapter_grouping = dict()
        adapter_grouping = {i: [] for i in range(num_adap)}
        senders = [sender for sender in cur_buffer.keys()]
        random.shuffle(senders)
        unassigned_client_num = len(senders)
        while unassigned_client_num > 0:
            num_finished = len(self.adapter_grouping)
            max_size = math.ceil(unassigned_client_num /
                                 (num_adap - num_finished))

            # step 1: Assign to the adapter where the clients
            # well performs
            for sender in senders:
                adap_idx, loss = next(cur_buffer[sender])
                while adap_idx not in adapter_grouping:
                    adap_idx, loss = next(cur_buffer[sender])
                adapter_grouping[adap_idx].append(sender)

            # step 2: Find the adapter with the most clients
            max_adap_idx_size = [0, 0]
            for adap_idx, candidates in adapter_grouping.items():
                if len(candidates) > max_adap_idx_size[1]:
                    max_adap_idx_size = [adap_idx, len(candidates)]

            # step 3: If the number of candidates is greater than
            # max_size, preserve the first max_size
            adap_idx = max_adap_idx_size[0]
            candidates = adapter_grouping[adap_idx][:max_size]

            # step 4: update the senders list, remove the selected
            # adapter from adapter_grouping
            senders = adapter_grouping[adap_idx][max_size:]
            self.adapter_grouping[adap_idx] = candidates
            adapter_grouping.pop(adap_idx)
            unassigned_client_num -= len(self.adapter_grouping[adap_idx])
            logger.info(f'Adapter {adap_idx} is done with the clients '
                        f'{self.adapter_grouping[adap_idx]}')

        # broadcast the new grouping info to all clients
        for adap_idx, receiver in self.adapter_grouping.items():
            self.comm_manager.send(
                Message(msg_type='set_active_adapter_idx',
                        sender=self.ID,
                        receiver=receiver,
                        state=self.state,
                        timestamp=self.cur_timestamp,
                        content=adap_idx))

        # resume the training based on the new group...
        self._start_new_training_round(skip_grouping=True)

        return True  # move_on_flag
