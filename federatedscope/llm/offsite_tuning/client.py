import gc
import logging

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import b64deserializer
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

logger = logging.getLogger(__name__)


class OffsiteTuningClient(Client):
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
        super(OffsiteTuningClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)
        if self._cfg.llm.offsite_tuning.llm_generated.use:
            train_set = self.data['train'].dataset
            overwrite_ratio = self._cfg.llm.offsite_tuning.llm_generated.ratio
            for i in range(int(len(train_set) * overwrite_ratio)):
                idx = train_set.dataset.indices[train_set.indices[i]]
                train_set.dataset.dataset.overwrite_by_llm(idx, self._model)

        if self._cfg.federate.mode == 'standalone' and \
                self._cfg.federate.share_local_model:
            # self.model is emulator_and_adapter, so we do nothing
            pass
        else:
            # Delete the stored client's model
            delattr(self, '_model')
            delattr(self, 'trainer')
            gc.collect()
            self.trainer = None

    def _register_default_handlers(self):
        super(OffsiteTuningClient, self)._register_default_handlers()
        self.register_handlers('emulator_and_adapter',
                               self.callback_funcs_for_emulator_and_adapter,
                               [None])

    def callback_funcs_for_emulator_and_adapter(self, message: Message):
        if self._cfg.federate.mode == 'standalone' and \
                self._cfg.federate.share_local_model:
            logger.info(f'Client {self.ID}: `share_local_model` mode '
                        f'enabled, emulator and adapter built from FedRunner.')
        else:
            logger.info(f'Client {self.ID}: Emulator and adapter received.')
            adapter_model = b64deserializer(message.content, tool='dill')

            # Define new model upon received
            self._model = adapter_model
            self.trainer = get_trainer(model=adapter_model,
                                       data=self.data,
                                       device=self.device,
                                       config=self._cfg,
                                       is_attacker=self.is_attacker,
                                       monitor=self._monitor)
