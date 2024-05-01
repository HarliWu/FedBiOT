import logging
import torch

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

        if self._cfg.llm.adapter.local_only:
            logger.warning("In local training mode, we will use all clients. "
                           "And we set the total round to 0 for one training "
                           "round only. ")

            self.sampler = None
            self.sample_client_num = client_num

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
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return aggregated_num
