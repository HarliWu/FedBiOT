import torch
import datasets
import json
import os
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


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

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval",
                                     "alpaca_eval",
                                     trust_remote_code=True)["eval"]
    generate_kwargs = dict(top_p=1.0,
                           temperature=0.0,
                           do_sample=False,
                           max_new_tokens=init_cfg.llm.max_new_token)

    eval_data_dict = []
    for example in tqdm(eval_set):
        record = {
            "instruction": example["instruction"],
            "input": None,
            "output": "",
            # "davinci_output": example["output"],
            "generator": init_cfg.federate.save_to,
            # "dataset": example["dataset"],
            # "datasplit": example["datasplit"]
        }
        if record["input"] is None:
            record["output"] = fschatbot.generate(
                [PROMPT_DICT["prompt_no_input"].format_map(record)],
                generate_kwargs)
        else:
            record["output"] = fschatbot.generate(
                [PROMPT_DICT["prompt_input"].format_map(record)],
                generate_kwargs)
        print(record)
        eval_data_dict.append(record)

    # save the evaluation result to a file
    eval_path = os.path.join(init_cfg.outdir, 'alpaca_eval.json')

    # create the directory if it does not exist
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)

    json.dump(eval_data_dict, open(eval_path, 'w'), indent=2)


if __name__ == "__main__":
    main()
