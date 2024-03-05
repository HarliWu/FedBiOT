import re
import torch
import sys
import transformers
import logging

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.llm.dataset.llm_dataset import LLMDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 10
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"

logger = logging.getLogger(__name__)


def gsm_demo_questions():
    question = []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")

    return question


def build_prompt(questions, n_shot=-1):
    demo_text = ""
    # demo_text += "The examples below are the questions for " \
    #              "large language model training.
    demo_text += "Generate a math question. " \
                 "Some examples are provided as follows. \n\n"

    if n_shot == -1:
        for question in questions:
            demo_text += "Q: " + question + "\n\n"
    else:
        for question in questions[-n_shot:]:
            demo_text += "Q: " + question + "\n\n"

    # demo += 'Generate a question for large language model training. \n\n'
    input_text_prompt = demo_text + "Q: "
    return input_text_prompt


class ArtifactDataset(FSChatBot):
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.add_special_tokens = True
        self.config = config

        if config.train.is_enable_half:
            self.model.to(torch.bfloat16).to(self.device)
        else:
            self.model.to(self.device)
        self.model = self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []

    def create(self, questions, num_samples, n_shot=N_SHOT):
        dataset = []

        for _ in range(num_samples):
            input_text = build_prompt(questions, n_shot)
            question = self.predict(input_text,
                                    use_history=False,
                                    use_prompt=False)
            logger.info('Question: ' + question)
            questions.append(question)

            answer = self.predict(question, use_history=False, use_prompt=True)
            logger.info("Answer: " + answer)
            logger.info('=' * 50)

            dataset.append({
                'instruction': question,
                'input': None,
                'output': answer,
                'category': None
            })

        return {
            'train': get_dataloader(LLMDataset(dataset, self.tokenizer),
                                    self.config),
            'val': None,
            'test': None
        }


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # init_cfg.freeze()

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    questions = gsm_demo_questions()

    for _ in range(10):
        input_text = build_prompt(questions, N_SHOT)
        generate_kwargs = dict(max_new_tokens=256,
                               top_p=0.95,
                               temperature=0.8,
                               early_stopping=True)
        # model_completion = fschatbot.generate(input_text, generate_kwargs)
        model_completion = fschatbot.predict(input_text,
                                             use_history=False,
                                             use_prompt=False)
        print(model_completion)
        questions.append(model_completion)

        answer = fschatbot.predict(model_completion,
                                   use_history=False,
                                   use_prompt=True)
        print(answer)
        print('=======================\n\n')
        # input_text = input_text + model_completion + "\n\n" + "Q: "


if __name__ == "__main__":
    main()
