import sys
import logging
import torch
import transformers
from transformers import pipeline, GenerationConfig
import os
import gc

transformers.logging.set_verbosity(40)

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT, DefaultToken
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.offsite_tuning.utils import \
    wrap_offsite_tuning_for_eval

logger = logging.getLogger(__name__)


def get_tokenizer(model_name, cache_dir, tok_len=128):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=tok_len,
        padding_side="left",
        use_fast=False,
    )

    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens


class FSChatBot(object):
    def __init__(self, config, use_raw=False):
        self.config = config

        self.device = f'cuda:{config.device}'
        self.add_special_tokens = True

        num_ckpt = config.federate.total_round_num // config.federate.save_freq
        self.prefix = ['final_'] + \
                      [str(i*config.federate.save_freq) + '_'
                       for i in range(num_ckpt, -1, -1)] + ['']
        self.dirname, self.filename = os.path.split(config.federate.save_to)
        print(self.prefix)
        if use_raw:
            self.use_raw_model()
        else:
            self.next_model()

    def use_raw_model(self):
        if hasattr(self, 'model'):
            delattr(self, 'model')
            gc.collect()
            torch.cuda.empty_cache()

        model_name, _ = self.config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, self.config.data.root,
                                          self.config.llm.tok_len)

        self.model = get_llm(self.config, device_map='auto')

        logger.info("will use raw model.")
        print("will use raw model.")

        self.model = self.model.to(self.device + 1)
        self.model = self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.max_history_len = self.config.llm.chat.max_history_len
        self.max_len = self.config.llm.chat.max_len
        self.history = []

    def next_model(self):
        if hasattr(self, 'model'):
            delattr(self, 'model')
            gc.collect()

        model_name, _ = self.config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, self.config.data.root,
                                          self.config.llm.tok_len)

        self.model = get_llm(self.config, device_map='auto')
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        logger.info(f'{model_name} default generation setting: '
                    f'{self.generation_config}')

        self.curpfx = None
        for pre in self.prefix:
            if os.path.exists(os.path.join(self.dirname, pre + self.filename)):
                self.curpfx = pre
                break

        # Load model from the checkpoints
        if self.curpfx is not None:
            ckpt_path = os.path.join(self.dirname, self.curpfx + self.filename)
            if self.config.llm.offsite_tuning.use:
                self.model = wrap_offsite_tuning_for_eval(
                    self.model, self.config, ckpt_path)
            else:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                if 'model' and 'cur_round' in ckpt:
                    self.model.load_state_dict(ckpt['model'])
                    logger.info(
                        f"Load with the model of Round {ckpt['cur_round']}")
                    print(f"Load with the model of Round {ckpt['cur_round']}")
                else:
                    self.model.load_state_dict(ckpt)
            logger.info(f'Model loads from the checkpoint {ckpt_path}')
            print(f'Model loads from the checkpoint {ckpt_path}')

            # remove the prefix up to the current one
            self.prefix = self.prefix[self.prefix.index(self.curpfx) + 1:]

        elif len(self.prefix) > 1:
            logger.info("will use raw model.")
            print("will use raw model.")
            self.prefix = []
            if self.config.llm.offsite_tuning.use:
                self.model = wrap_offsite_tuning_for_eval(
                    self.model, self.config)
        else:
            raise ValueError('No more model is able to us')

        self.model.to('cuda:0')
        self.model = self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # # Create the generation pipeline
        # self.generation_pipe = pipeline('text-generation',
        #                                 model=self.model,
        #                                 tokenizer=self.tokenizer,
        #                                 device_map='auto',
        #                                 trust_remote_code='True')

        self.max_history_len = self.config.llm.chat.max_history_len
        self.max_len = self.config.llm.chat.max_len
        self.history = []

    def _build_prompt(self, input_text):
        source = {'instruction': input_text}
        return PROMPT_DICT['prompt_no_input'].format_map(source)

    def predict(self, input_text, use_history=True, use_prompt=True):
        if use_prompt:
            input_text = self._build_prompt(input_text)
        text_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = []
        if use_history:
            for history_ctx in self.history[-self.max_history_len:]:
                input_ids.extend(history_ctx)
        else:
            input_ids.extend(text_ids)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        response = self.model.generate(input_ids=input_ids,
                                       max_new_tokens=self.max_len,
                                       num_beams=4,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True,
                                       temperature=0.2)

        self.history.append(response[0].tolist())
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        return response_tokens

    @torch.no_grad()
    def generate(self,
                 input_texts: list[str],
                 generate_kwargs={}) -> list[list[str]]:
        input_text_tokens = self.tokenizer(
            input_texts,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to("cuda:0")

        output_ids = self.model.generate(**input_text_tokens,
                                         **generate_kwargs)
        responses = self.tokenizer.batch_decode(output_ids,
                                                skip_special_tokens=True,
                                                ignore_tokenization_space=True)

        response_map = [[] for _ in input_texts]
        for res in responses:
            for idx, input_text in enumerate(input_texts):
                if input_text in res:
                    gen_res = res.replace(input_text, "").strip()
                    response_map[idx].append(gen_res.replace("</s>", ""))
                    # response_map[idx].append(
                    #     " " + gen_res.replace("</s>", ""))
                    break

        return response_map

    def clear(self):
        self.history = []


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    chat_bot = FSChatBot(init_cfg)
    welcome = "Welcome to FSChatBot, " \
              "`clear` to clear history, " \
              "`quit` to end chat."
    print(welcome)
    while True:
        input_text = input("\nUser:")
        if input_text.strip() == "quit":
            break
        if input_text.strip() == "clear":
            chat_bot.clear()
            print(welcome)
            continue
        print(f'\nFSBot: {chat_bot.predict(input_text)}')


if __name__ == "__main__":
    main()
