# ref: https://github.com/mit-han-lab/offsite-tuning/blob/
# main/offsite_tuning/eval_harness.py

import torch

from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import evaluator, tasks
import lm_eval.tasks.hellaswag as hellaswag
from lm_eval.api.registry import ALL_TASKS

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot


class LMEvaluator(HFLM):
    def __init__(self, fschatbot: FSChatBot):
        super().__init__(pretrained=fschatbot.model.model,
                         backend="causal",
                         tokenizer=fschatbot.tokenizer,
                         device='cuda')

        self._rank = 0

    def loglikelihood(self, requests):
        for i in range(20):
            print(requests[i].request_type)
        super().loglikelihood(requests)


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

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    task_manager = TaskManager()
    # print(task_manager.all_tasks)
    task_names = task_manager.match_tasks(['hellaswag'])
    # Load to the evaluator
    results = evaluator.simple_evaluate(
        model=LMEvaluator(fschatbot),
        tasks=task_names,
        write_out=False,
        device='cuda:0',
        batch_size='auto',
        max_batch_size=50,
    )
    print(results)


if __name__ == "__main__":
    main()
