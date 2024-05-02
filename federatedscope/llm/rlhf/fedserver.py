import os
import copy
import logging

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data import ClientData
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.workers.server import Server
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.rlhf.standalone_training import \
    RLHF_finetuning
from federatedscope.llm.trainer.reward_choice_trainer import \
    RewardChoiceTrainer
from federatedscope.llm.dataset.llm_dataset import LLMDataset

logger = logging.getLogger(__name__)


def get_prompt(config):
    dataset_name, _ = config.data.type.split('@')

    if 'reddit-tldr' in dataset_name.lower():
        from federatedscope.llm.dataloader.reddit_tldr import \
            TLDR_PROMPT_DICT
        generation_prompt = TLDR_PROMPT_DICT['summary']
        selector_prompt = TLDR_PROMPT_DICT['summary_cmp']

    return generation_prompt, selector_prompt


def build_cfg_for_policy_training(config):
    new_cfg = global_cfg.clone()
    new_cfg.merge_from_file(config.llm.fedrlhf.config_file)
    new_cfg.llm.rlhf = True

    # TODO: might generate extra cfg file, delete
    new_cfg.freeze(save=False)
    return new_cfg


def build_cfg_for_policy_reward_alignment(config):
    new_cfg = copy.deepcopy(config)
    new_cfg.defrost()

    new_cfg.train.local_update_steps = \
        config.llm.fedrlhf.train.local_update_steps
    new_cfg.train.batch_or_epoch = \
        config.llm.fedrlhf.train.batch_or_epoch

    # # Overwrite `config.data` with
    # # `config.llm.offsite_tuning.emu_align.data`
    # for key, value in \
    #         new_cfg.llm.offsite_tuning.emu_align.data.items():
    #     if key.startswith('__') or (key not in new_cfg.data.keys()):
    #         continue
    #     setattr(new_cfg.data, f'{key}', value)
    # Used for data translator
    new_cfg.federate.client_num = 1

    # TODO: might generate extra cfg file, delete
    new_cfg.freeze(save=False)
    return new_cfg


class RewardServer(Server):
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
        super(RewardServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        # Load the tokenizer of the reward model
        model_name, _ = self._cfg.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, self._cfg.data.root,
                                          self._cfg.llm.tok_len)
        _, save_to = os.path.split(self._cfg.federate.save_to)
        self.save_to_prefix = save_to.replace('.ckpt', '')

        # Build the alignment config between reward and policy model
        self.align_cfg = build_cfg_for_policy_reward_alignment(self._cfg)

        # Load the policy model (LLM)
        policy_config = build_cfg_for_policy_training(config)
        policy_model_name, _ = policy_config.model.type.split('@')
        policy_model = get_llm(policy_config,
                               device_map='auto',
                               load_from_prev_ckpt=True)
        logger.info('Successfully load the policy model...')
        policy_tokenizer, _ = get_tokenizer(policy_model_name,
                                            policy_config.data.root,
                                            policy_config.llm.tok_len)
        self.policy_trainer = RLHF_finetuning(policy_model, policy_tokenizer,
                                              policy_config, self.model,
                                              self.tokenizer)

        if self._cfg.llm.fedrlhf.pretrained:
            logger.info(
                'Pretrained the selector model with the policy model...')
            # Policy model acts as a reward model
            list_train_dict = self.policy_trainer.dpo_better_response()
            self._dpo_alignment(list_train_dict)

        self.policy_trainer.model.cpu()

    def _dpo_alignment(self, list_dpo_data_dict):
        # convert the choice from 0,1 to A,B
        for sample in list_dpo_data_dict:
            sample['choice'] = " " + chr(sample['choice'] + ord("A"))
        # load DPO dataset
        _, selector_prompt = get_prompt(self.align_cfg)
        train_dataset = LLMDataset(list_dpo_data_dict,
                                   self.tokenizer,
                                   prompt_input=selector_prompt,
                                   prompt_no_input=selector_prompt,
                                   output_tag='choice')
        data = ClientData(self.align_cfg, train_dataset, None, None)
        # create a trainer for alignment
        trainer = RewardChoiceTrainer(self.model,
                                      data,
                                      self.device,
                                      self.align_cfg,
                                      only_for_eval=False,
                                      monitor=Monitor(self.align_cfg,
                                                      monitored_object=self))
        trainer.train()
        logger.info('Reward model has aligned with the policy model...')

    def _start_new_training_round(self, aggregated_num=0):
        if self.state % self._cfg.llm.fedrlhf.frequency == 0:
            self.policy_trainer.model.sharding()
            self.policy_trainer.train(
                saveto=f'{self.state}_{self.save_to_prefix}')
            list_train_dict = self.policy_trainer.dpo_better_response()
            self._dpo_alignment(list_train_dict)
            self.policy_trainer.model.cpu()

        # Broadcast the model and start the training
        super()._start_new_training_round(aggregated_num)
