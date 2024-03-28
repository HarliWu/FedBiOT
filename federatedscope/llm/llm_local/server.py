import logging
import copy
import torch

from federatedscope.core.message import Message
from federatedscope.register import register_worker
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.utils import add_prefix_to_path
from federatedscope.core.workers.server import Server
from federatedscope.llm.offsite_tuning.utils import \
    build_cfg_for_alignment, convert_layers_train_state

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
            self.total_round_num = 0

    def callback_funcs_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        logger.info(f'{round}, {sender}, {content}')

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            if isinstance(content[1], list):  # multiple model
                sample_size = content[0]
                quant_model = [
                    symmetric_uniform_dequantization(x) for x in content[1]
                ]
            else:
                sample_size = content[0]
                quant_model = symmetric_uniform_dequantization(content[1])
            content = (sample_size, quant_model)

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
            # TODO: Save the client models to files
            logger.info(content)

        move_on_flag = self.check_and_move_on()

        return move_on_flag

    # def broadcast_model_para(self,
    #                          msg_type='model_para',
    #                          sample_client_num=-1,
    #                          filter_unseen_clients=True):
    #     pass
