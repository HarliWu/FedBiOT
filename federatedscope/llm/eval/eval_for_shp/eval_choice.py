import re
import logging
import torch
from torch.utils.data import DataLoader
import os
from transformers import GenerationConfig
from tqdm import tqdm
import numpy as np

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.dataloader.shp import \
    load_shp_cmp_dataset_by_choice
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader import get_tokenizer, LLMDataCollator
from federatedscope.llm.dataset.llm_dataset import DefaultToken

logger = logging.getLogger(__name__)


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

        test_batches.set_postfix({
            'correct': correct,
            'total': total,
            'rate': '{:.2f}%'.format(correct / total * 100)
        })

    return expected_choice, actual_choice, correct, total


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
    # model = model.merge_and_unload()

    # load dataset
    data_root = os.path.join(init_cfg.data.root, 'shp')
    _, _, test_dataset = load_shp_cmp_dataset_by_choice(data_root=data_root,
                                                        tokenizer=tokenizer)

    # list all choices
    choices = []
    for choice in init_cfg.trainer.choices:
        choices.append(tokenizer(f'{choice}')['input_ids'][-1])

    # Print result to a text file
    results_display = open(os.path.join(init_cfg.outdir, 'test_results.txt'),
                           'w')
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

    # Print some samples
    for batch_idx, data_batch in enumerate(dataloader):
        input_ids = data_batch["input_ids"].to('cuda:0')

        idx = batch_idx * input_ids.size(0)

        # extract the first result
        first_input = tokenizer.decode(
            input_ids[0][input_ids[0] != tokenizer.pad_token_id],
            skip_special_tokens=True)
        first_expected_choice = chr(expected_choice[idx] + ord('A'))
        first_actual_choice = chr(actual_choice[idx] + ord('A'))

        # format the input
        sample = {
            "post": "POST: ",
            "output_A": "SUMMARY A: ",
            "output_B": "SUMMARY B: ",
        }
        for seg in first_input.split("### "):
            for key in sample.keys():
                if sample[key] in seg:
                    sample[key] = seg.replace(sample[key], '')
                    break

        # write the first input and results to file
        results_display.write(f'Post:\n{sample["post"]}\n\n'
                              f'Summary A:\n{sample["output_A"]}\n\n'
                              f'Summary B:\n{sample["output_B"]}\n\n'
                              f'Human selected: {first_expected_choice}\n\n'
                              f'Model-choice: {first_actual_choice}\n\n')
        results_display.write('==========================\n\n')
        results_display.flush()

    results_display.close()


if __name__ == "__main__":
    main()
