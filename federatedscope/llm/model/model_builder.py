import os

from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.core.configs.config import global_cfg
import torch

import logging

logger = logging.getLogger(__name__)


def get_model_from_huggingface(model_name, config, **kwargs):
    from transformers import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    if config.train.is_enable_half:
        kwargs['torch_dtype'] = torch.bfloat16

    if config.model.llm_type == 'SequenceClassification':
        from transformers import AutoModelForSequenceClassification
        if len(config.model.llm_kwargs) > 0:
            kwargs.update(config.model.llm_kwargs[0])
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_model_from_modelscope(model_name, config, **kwargs):
    from modelscope import AutoModelForCausalLM

    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config, load_from_prev_ckpt=False, **kwargs):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')

    if config.model.load_from_local_pretrained_fs_config != '':
        # load model from local pretrained model
        pretrained_cfg = global_cfg.clone()
        pretrained_cfg.merge_from_file(
            config.model.load_from_local_pretrained_fs_config)
        assert pretrained_cfg.model.type.split('@')[0] == model_name, \
            'Two models cannot match. Failed to load from pretrained.'
        pretrained_model = get_llm(pretrained_cfg, **kwargs)
        if config.model.load_from_local_pretrained_model_path != '':
            path = config.model.load_from_local_pretrained_model_path
            ckpt = torch.load(path, map_location='cpu')
            logger.info('Successfully import the pretrained model '
                        f'from the checkpoint {path}. ')
            pretrained_model.load_state_dict(ckpt['model'])
        model = pretrained_model.merge_and_unload()
        logger.info(f'Merge and unload to {type(model)}...')
    elif model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config,
                                           **kwargs)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name,
                                          config=config,
                                          **kwargs)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)
    if model_config.llm_type == 'SequenceClassification':
        model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        if model_config.llm_type != 'SequenceClassification':
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    if config.llm.adapter.use and config.llm.adapter.local_only:
        model.append_adapters(adapter_names=[
            f'Adapter_{i+1}' for i in range(config.federate.client_num)
        ])

    if config.llm.adapter.count > 1:
        model.append_adapters(adapter_names=[
            f'Adapter_{i}' for i in range(config.llm.adapter.count)
        ])

    if load_from_prev_ckpt:
        # Here we load from the most recent one
        num_ckpt = config.federate.total_round_num // config.federate.save_freq
        prefix = ['final_'] + \
            [str(i*config.federate.save_freq) + '_'
             for i in range(num_ckpt, -1, -1)] + ['']
        dirname, filename = os.path.split(config.federate.save_to)
        for pre in prefix:
            ckpt_path = os.path.join(dirname, pre + filename)
            logger.info(f'Attempt to load from {ckpt_path}')
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                logger.info(f'Model of Round {ckpt["cur_round"]} loads '
                            f'from the checkpoint {ckpt_path}')
                break

    return model
