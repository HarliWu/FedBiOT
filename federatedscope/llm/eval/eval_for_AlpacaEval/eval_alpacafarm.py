import torch
from torch.utils.data import DataLoader

import datasets
import json
from tqdm import tqdm
import os
import numpy as np

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader import get_tokenizer, LLMDataCollator

PROMPT = "### User: {instruction}\n\n### Assistant:"
PROMPT_SHP = ("Below is an instruction that describes a task. "
              "Write a response that appropriately completes the request.\n\n"
              "### Instruction:\n{instruction}\n\n"
              "### Response:")
PROMPT_CMP = ("Below is a query followed by two responses. Pick a "
              "helpful response that is precise, concise, and casual. "
              "State your choice with a single capital letter, "
              "i.e., \"A\" if RESPONSE A is better, "
              "\"B\" if RESPONSE B is better.\n\n"
              "### QUERY: {instruction}\n"
              "### RESPONSE A: {output_A}\n"
              "### RESPONSE B: {output_B}\n"
              "### YOUR CHOICE:")


def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, predicted, predicted.eq(new_labels).sum().item()


def evaluation(model, dataloader, choices):
    test_batches = tqdm(dataloader)
    correct, total = 0, 0
    expected_choice, actual_choice = [], []
    for data_batch in test_batches:
        input_ids = data_batch["input_ids"].to('cuda:0')
        labels = data_batch["labels"].to('cuda:0')
        attention_mask = data_batch["attention_mask"].to('cuda:0')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # calculate the correctness
        new_labels, predicted, batch_correct = \
            cal_acc(outputs.logits, labels, choices)

        # collect the expected and actual results
        expected_choice += new_labels.tolist()
        actual_choice += predicted.tolist()

        # Display the final result on screen
        total = total + new_labels.size(0)
        correct = correct + batch_correct

        # print(correct, total, new_labels, labels)
        test_batches.set_postfix({
            'correct': correct,
            'total': total,
            'rate': '{:.2f}%'.format(correct / total * 100)
        })

    return expected_choice, actual_choice, correct, total


@torch.no_grad()
def eval_for_generation(init_cfg, prompt=PROMPT):
    """
    win rate compared with GPT-generated response
    """
    # load dataset
    data = datasets.load_dataset("tatsu-lab/alpaca_farm",
                                 name="alpaca_farm_evaluation",
                                 split="eval")

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)
    generate_kwargs = dict(top_p=1.0,
                           temperature=0.0,
                           do_sample=False,
                           max_new_tokens=init_cfg.llm.max_new_token)

    eval_data_dict = []
    for example in tqdm(data):
        if example["input"] is None or example["input"] == "":
            new_ins = example["instruction"] + "\n\n" + example["input"]
            record = {
                "instruction": new_ins,
                "output": "",
                "generator": init_cfg.federate.save_to,
                "dataset": example["dataset"],
                "datasplit": example["datasplit"]
            }
        else:
            record = {
                "instruction": example["instruction"],
                "output": "",
                "generator": init_cfg.federate.save_to,
                "dataset": example["dataset"],
                "datasplit": example["datasplit"]
            }
        record["output"] = fschatbot.generate(PROMPT.format_map(record),
                                              generate_kwargs)
        eval_data_dict.append(record)

    # save the evaluation result to a file
    eval_path = os.path.join(init_cfg.outdir,
                             f'{fschatbot.curpfx}_alpaca_eval.txt')
    json.dump(eval_data_dict, open(eval_path, 'w'), indent=2)

    return eval_data_dict


@torch.no_grad()
def eval_for_agreement(init_cfg, label, prompt=PROMPT_CMP):
    # get model and tokenizer
    model_name, _ = init_cfg.model.type.split('@')
    model = get_llm(init_cfg, device_map='auto')
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root,
                                 init_cfg.llm.tok_len)

    # load model from checkpoint
    num_ckpt = \
        init_cfg.federate.total_round_num // init_cfg.federate.save_freq
    prefix = ['final_'] + \
             [str(i*init_cfg.federate.save_freq) + '_'
              for i in range(num_ckpt, -1, -1)] + ['']
    dirname, filename = os.path.split(init_cfg.federate.save_to)
    for pre in prefix:
        print(os.path.join(dirname, pre + filename))
        if os.path.exists(os.path.join(dirname, pre + filename)):
            ckpt_path = os.path.join(dirname, pre + filename)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            print(f'Model of Round {ckpt["cur_round"]} loads '
                  f'from the checkpoint {ckpt_path}')
            break

    # list all choices
    choices = []
    for choice in init_cfg.trainer.choices:
        choices.append(tokenizer(f'{choice}')['input_ids'][-1])

    # load dataset
    list_data_dict = []
    data = datasets.load_dataset("tatsu-lab/alpaca_farm",
                                 name=f"alpaca_{label}_preference",
                                 split="preference")
    for example in data:
        if example["input"] is None or example["input"] == "":
            new_ins = example["instruction"] + "\n\n" + example["input"]
            record = {
                "instruction": new_ins,
                "output_A": example["output_1"],
                "output_B": example["output_2"],
                "choice": " " + chr(example["preference"] - 1 + ord("A")),
            }
        else:
            record = {
                "instruction": example["instruction"],
                "output_A": example["output_1"],
                "output_B": example["output_2"],
                "choice": " " + chr(example["preference"] - 1 + ord("A")),
            }
        list_data_dict.append(record)

    test_dataset = LLMDataset(list_data_dict,
                              tokenizer,
                              prompt_input=prompt,
                              prompt_no_input=prompt,
                              output_tag='choice')

    # Print result to a text file
    results_display = open(
        os.path.join(init_cfg.outdir, f'{label}_test_results.txt'), 'w')
    dataloader = DataLoader(dataset=test_dataset,
                            batch_size=10,
                            shuffle=False,
                            collate_fn=LLMDataCollator(tokenizer=tokenizer))

    expected_choice, actual_choices = [], []
    if init_cfg.llm.adapter.local_only:
        for i in range(init_cfg.federate.client_num):
            model.set_active_adapter(f'Adapter_{i+1}')
            model.eval()
            expected_choice, actual_choice, correct, total = \
                evaluation(model, dataloader, choices)
            acc = correct / total * 100
            results_display.write(
                f'Client {i+1} (Adapter_{i+1}): \n'
                f'Correct: {correct}; Total: {total}; Accuracy: {acc}%\n\n')
            actual_choices.append(actual_choice)
    elif init_cfg.llm.adapter.count > 1:
        for i in range(init_cfg.llm.adapter.count):
            model.set_active_adapter(f'Adapter_{i}')
            model.eval()
            expected_choice, actual_choice, correct, total = \
                evaluation(model, dataloader, choices)
            acc = correct / total * 100
            results_display.write(
                f'Adapter_{i}: \n'
                f'Correct: {correct}; Total: {total}; Accuracy: {acc}%\n\n')
            actual_choices.append(actual_choice)
    else:
        model.eval()
        expected_choice, actual_choice, correct, total = \
            evaluation(model, dataloader, choices)
        actual_choices.append(actual_choice)

    # Choose the indices with the most votes / or the actual choice
    array = np.array(actual_choices).T
    actual_choice = [np.bincount(array[i]).argmax() for i in range(len(array))]

    # Display the overall results
    indicator = (np.array(expected_choice) == np.array(actual_choice))
    correct, total = np.sum(indicator), len(indicator)
    acc = correct / total * 100
    results_display.write(
        f'Overall: \n'
        f'Correct: {correct}; Total: {total}; Accuracy: {acc}%\n\n'
        '==========================\n\n')


@torch.no_grad()
def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    init_cfg.freeze()

    if 'rewardchoice' in init_cfg.federate.save_to:
        # eval for agreement rate
        eval_for_agreement(init_cfg, 'human')
        eval_for_agreement(init_cfg, 'gpt4')
    else:
        # eval for win rate
        if 'shp' in init_cfg.federate.save_to:
            eval_for_generation(init_cfg, prompt=PROMPT_SHP)
        else:
            eval_for_generation(init_cfg)


if __name__ == "__main__":
    main()
