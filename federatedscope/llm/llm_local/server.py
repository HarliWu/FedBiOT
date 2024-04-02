import logging

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
            merged_adapter = dict()

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    _, model_param = train_msg_buffer[client_id]
                    merged_adapter.update(model_param)
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    merged_adapter.update(model_para_multiple[model_idx])

            msg_list = [(1, merged_adapter)]

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
