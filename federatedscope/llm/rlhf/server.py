import logging
import os
import random
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data import ClientData
from federatedscope.llm.dataloader import LLMDataCollator
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, LLMComparisonDataset
from federatedscope.llm.trainer.reward_trainer import RewardTrainer
from federatedscope.core.auxiliaries.utils import add_prefix_to_path

logger = logging.getLogger(__name__)


@torch.no_grad()
def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    # print(new_logits)
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, new_logits, predicted, predicted.eq(
        new_labels).sum().item()


def get_rlhf_dataset(config):
    dataset_name, _ = config.data.type.split('@')

    if dataset_name.lower() == 'reddit-tldr-rlhf':
        from federatedscope.llm.dataloader.reddit_tldr import \
            load_human_finetuning_dataset, TLDR_PROMPT_DICT
        list_train_dict, _, _ = \
            load_human_finetuning_dataset(config.data.root,
                                          tokenizer=None,
                                          rlhf=True,
                                          max_num_test=1000,
                                          raw_no_prompt=True)
        generation_prompt = TLDR_PROMPT_DICT['summary']
        selector_prompt = TLDR_PROMPT_DICT['summary_cmp']

    return list_train_dict, generation_prompt, selector_prompt


class RLHF_finetuning:
    """
    Implementation of RLHF server
    """
    def __init__(self,
                 model,
                 tokenizer,
                 config=None,
                 selector_model=None,
                 selector_tokenizer=None,
                 device='cpu',
                 **kwargs):
        # obtain RLHF input data
        self.list_train_dict, self.generation_prompt, self.selector_prompt = \
            get_rlhf_dataset(config)

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.selector_model = selector_model
        self.selector_tokenizer = selector_tokenizer
        self.device = device
        self._monitor = Monitor(config, monitored_object=self)

    def train(self):
        fp = os.path.join(self.config.data.root, 'generated_rlhf_data.jsonl')

        if os.path.exists(fp):
            list_train_dict = json.load(open(fp, "r"))

        else:
            # generate the output
            list_train_dict = self._generate_pairwise_data(
                self.list_train_dict,
                self.model,
                self.tokenizer,
                self.generation_prompt,
                max_new_tokens=self.config.llm.max_new_token)

            # save the data to a file
            json.dump(list_train_dict, open(fp, "w"))

        # choose the better one based on the given output
        list_train_dict = self._choose_better_response(list_train_dict,
                                                       self.selector_model,
                                                       self.selector_tokenizer,
                                                       self.selector_prompt)

        # load comparison dataset
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            self.tokenizer,
            prompt_input=self.generation_prompt,
            prompt_no_input=self.generation_prompt,
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        data = ClientData(self.config, train_dataset, None, None)

        # create DPO trainer
        self.trainer = RewardTrainer(self.model,
                                     data,
                                     self.device,
                                     self.config,
                                     only_for_eval=False,
                                     monitor=self._monitor)

        # start training
        for r in range(self.config.federate.total_round_num):
            logger.info(f'----------- Starting a new training round '
                        f'(Round #{r}) -------------')
            sample_size, model_para_all, results = self.trainer.train()
            train_log_res = self._monitor.format_eval_res(results,
                                                          rnd=r,
                                                          role='Server',
                                                          return_raw=True)
            logger.info(train_log_res)
            # Save the checkpoint
            path = add_prefix_to_path(f'{r}_', self.config.federate.save_to)
            self.model.save_model(path=path, state=r)

    def _generate_pairwise_data(self,
                                list_data_dict,
                                model,
                                tokenizer,
                                prompt,
                                max_new_tokens=60):
        generate_kwargs = dict(
            top_p=1.0,
            temperature=1.0,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            num_return_sequences=2,
        )

        for data in tqdm(list_data_dict):
            input_text = prompt.format_map(data)
            input_text_tokens = tokenizer(
                input_text,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = input_text_tokens.input_ids.to('cuda:0')
            attention_mask = input_text_tokens.attention_mask.to('cuda:0')

            output_ids = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        **generate_kwargs)

            response = []
            for i in range(output_ids.shape[0]):
                response.append(
                    self.tokenizer.decode(output_ids[i][input_ids.shape[1]:],
                                          skip_special_tokens=True,
                                          ignore_tokenization_space=True))
                if response[-1].startswith(" ") is False:
                    response[-1] = " " + response[-1]

            data['output_A'] = response[0]
            data['output_B'] = response[1]

        return list_data_dict

    def _choose_better_response(self, list_data_dict, model, tokenizer,
                                prompt):
        choices = [tokenizer(f'{c}')['input_ids'][-1] for c in ['A', 'B']]

        for sample in list_data_dict:
            sample['fake_choice'] = random.choice([" A", " B"])

        token_dataset = LLMDataset(list_data_dict,
                                   tokenizer,
                                   prompt_input=prompt,
                                   prompt_no_input=prompt,
                                   output_tag='fake_choice')
        dataloader = DataLoader(
            dataset=token_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=LLMDataCollator(tokenizer=tokenizer))

        predicted_indices = []
        for idx, data_batch in enumerate(tqdm(dataloader)):
            input_ids = data_batch["input_ids"].to('cuda:0')
            labels = data_batch["labels"].to('cuda:0')
            attention_mask = data_batch["attention_mask"].to('cuda:0')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, _, predicted, _ = cal_acc(outputs.logits, labels, choices)
            predicted_indices += predicted.tolist()

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample['choice'] = choice
            sample.pop('fake_choice', None)

        return list_data_dict
