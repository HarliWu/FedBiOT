import gc
import logging

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import b64deserializer
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

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
        logger.info(self.trainer.ctx.model.get_active_adapter())

        return super().callback_funcs_for_model_para(message)
