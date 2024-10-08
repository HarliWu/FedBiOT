import gc
import os
import copy
import logging
import torch
import torch.nn as nn

from transformers import (OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM,
                          LlamaForCausalLM)
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.offsite_tuning.kd_trainer import KDTrainer
from federatedscope.core.auxiliaries.data_builder import get_data

logger = logging.getLogger(__name__)


def add_prologue(module, prologue):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
    module.old_forward = module.forward
    module.prologue = prologue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            self.input_args = args
            self.input_kwargs = kwargs
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            args = (x, ) + args[1:]
            return self.old_forward(*args, **kwargs)

        return lambda_forward

    module.forward = new_forward(module)
    return module


def add_epilogue(module, epilogue):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)

            if isinstance(output, tuple):
                output = (x, ) + output[1:]
            else:
                output = x

            self.cached_output = x
            return output

        return lambda_forward

    module.forward = new_forward(module)
    return module


def get_layers(adapter_model):
    """
    Modified from the official implementation:
    https://github.com/mit-han-lab/offsite-tuning/tree/main
    """
    try:
        return adapter_model.layers
    except:
        return adapter_model.model.transformer.h

    # if isinstance(adapter_model.model, OPTForCausalLM):
    #     layers = adapter_model.model.model.decoder.layers
    # elif isinstance(adapter_model.model, GPT2LMHeadModel):
    #     layers = adapter_model.model.transformer.h
    # elif isinstance(adapter_model.model, BloomForCausalLM):
    #     layers = adapter_model.model.transformer.h
    # elif isinstance(adapter_model.model, LlamaForCausalLM):
    #     layers = adapter_model.model.model.layers
    # else:
    #     # TODO: support more LLM
    #     logger.warning(f'Model {type(adapter_model.model)} not support, '
    #                    f'use default setting.')
    #     layers = adapter_model.model.transformer.h

    # return layers


def set_layers(adapter_model, layers, emu_l=0, emu_r=-1):
    # print(adapter_model)
    # print(adapter_model.layers)
    # print(layers)

    adapter_model.set_layers(layers)

    # if isinstance(adapter_model.model, OPTForCausalLM):
    #     adapter_model.model.model.decoder.layers = layers
    # elif isinstance(adapter_model.model, GPT2LMHeadModel):
    #     adapter_model.model.transformer.h = layers
    # elif isinstance(adapter_model.model, BloomForCausalLM):
    #     adapter_model.model.transformer.h = layers
    # elif isinstance(adapter_model.model, LlamaForCausalLM):
    #     adapter_model.model.model.layers = layers
    # else:
    #     # TODO: support more LLM
    #     logger.warning(f'Model {type(adapter_model.model)} not support, '
    #                    f'use default setting.')
    #     adapter_model.model.transformer.h = layers
    adapter_model.student = layers[emu_l:emu_r]
    adapter_model.adapter = layers[:emu_l] + layers[emu_r:]
    add_prologue(adapter_model.student[0], None)
    add_epilogue(adapter_model.student[-1], None)
    adapter_model.student_l = adapter_model.student[0]
    adapter_model.student_r = adapter_model.student[-1]
    return adapter_model


def model_drop_layer(layers, drop_ratio=0.5, **kwargs):
    new_model = nn.ModuleList()
    num_new_layers = round(len(layers) * (1 - drop_ratio))

    stride = (len(layers) - 1) / (num_new_layers - 1)
    new_model_maps = []

    for i in range(num_new_layers):
        idx = int(i * stride)
        logger.info(f"Adding layer {idx} to emulator.")
        print(f"Adding layer {idx} to emulator.")
        new_model.append(layers[idx])
        new_model_maps.append(idx)

    return new_model, new_model_maps


def model_pruning(model, ratio=0.5, **kwargs):
    raise NotImplementedError


def model_quantization(model, bits, **kwargs):
    raise NotImplementedError


def model_distillation(model, **kwargs):
    raise NotImplementedError


COMP_FUNC_MAPPING = {
    'drop_layer': model_drop_layer,
    'pruning': model_pruning,
    'quantization': model_quantization,
    'distillation': model_distillation
}


def generate_adap_model(model: AdapterModel, offsite_tuning_cfg):
    if offsite_tuning_cfg.strategy in COMP_FUNC_MAPPING.keys():
        compress_strategy = offsite_tuning_cfg.strategy
        emulator_l = offsite_tuning_cfg.emu_l
        emulator_r = offsite_tuning_cfg.emu_r
        emu_align = offsite_tuning_cfg.emu_align.use
        offsite_tuning_kwargs = offsite_tuning_cfg.kwargs[0]
        adap_model = generate_emulator_and_adapter(
            model,
            strategy=compress_strategy,
            emulator_l=emulator_l,
            emulator_r=emulator_r,
            emulator_alignment=emu_align,
            **offsite_tuning_kwargs)
    else:
        raise NotImplementedError

    # Use the following assertion to ensure that
    # model and adap_model have required attributes
    assert hasattr(model, 'teacher')  # a.k.a. raw emulator
    assert hasattr(model, 'adapter')
    assert hasattr(adap_model, 'student')  # a.k.a. emulator
    assert hasattr(adap_model, 'adapter')
    return adap_model


def generate_emulator_and_adapter(model: AdapterModel,
                                  strategy='drop_layer',
                                  emulator_l=0,
                                  emulator_r=1000,
                                  emulator_alignment=False,
                                  **kwargs):
    layers = get_layers(model)
    l, r = max(emulator_l, 0), min(emulator_r, len(layers) - 1)

    # make all parameters on raw model untrainable
    for module in model.modules():
        module.requires_grad_(False)

    # Set teacher model
    model.teacher = layers[l:r]  # Ref for old model
    model.adapter = layers[:l] + layers[r:]

    emulator, emulator_maps = \
        COMP_FUNC_MAPPING[strategy](model.teacher, **kwargs)

    emulator_and_adapter = nn.ModuleList()

    # Adapter before Emulator, make it trainable
    for idx in range(l):
        emulator_and_adapter.append(layers[idx])
    emu_l = l

    # Emulator
    for idx in range(len(emulator)):
        emulator_and_adapter.append(emulator[idx])
    emu_r = l + len(emulator)

    # Adapter after Emulator, make it trainable
    for idx in range(r, len(layers)):
        emulator_and_adapter.append(layers[idx])

    new_model = copy.deepcopy(model)
    new_emulator_and_adapter = copy.deepcopy(emulator_and_adapter)
    # Set student model
    new_model = set_layers(new_model, new_emulator_and_adapter, emu_l, emu_r)
    new_model.teacher_model_mapping = emulator_maps
    # make the adapter trainable on clients' models
    convert_layers_train_state(
        new_model.adapter,
        name_pattern=new_model.trainable_param_name_pattern,
        is_trainable=True)
    # make the emulator untrainable on clients' models
    convert_layers_train_state(new_model.student, is_trainable=False)

    gc.collect()
    torch.cuda.empty_cache()

    return new_model


def convert_layers_train_state(layers, name_pattern=None, is_trainable=True):
    if is_trainable:
        for layer in layers:
            for name, param in layer.named_parameters():
                if name_pattern is None or name_pattern in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    else:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


def build_cfg_for_alignment(config):
    new_cfg = copy.deepcopy(config)
    new_cfg.defrost()

    # # Overwrite `config.train` with
    # # `config.llm.offsite_tuning.emu_align.train`
    # for key, value in \
    #         new_cfg.llm.offsite_tuning.emu_align.train.optimizer.items():
    #     if key.startswith('__'):
    #         continue
    #     setattr(new_cfg.train.optimizer, f'{key}', value)
    new_cfg.train.local_update_steps = \
        config.llm.offsite_tuning.emu_align.train.local_update_steps
    new_cfg.train.batch_or_epoch = \
        config.llm.offsite_tuning.emu_align.train.batch_or_epoch

    # Overwrite `config.data` with
    # `config.llm.offsite_tuning.emu_align.data`
    for key, value in \
            new_cfg.llm.offsite_tuning.emu_align.data.items():
        if key.startswith('__') or (key not in new_cfg.data.keys()):
            continue
        setattr(new_cfg.data, f'{key}', value)
    # Used for data translator
    new_cfg.federate.client_num = 1

    # TODO: might generate extra cfg file, delete
    new_cfg.freeze()
    return new_cfg


def align_student_with_teacher(raw_model, adap_model, cfg, device, monitor):
    does_train_emulator = True
    if cfg.llm.offsite_tuning.emu_align.restore_from != '':
        try:
            if not os.path.exists(
                    cfg.llm.offsite_tuning.emu_align.restore_from):
                logger.warning(
                    f'Invalid `emu_align.restore_from`:'
                    f' {cfg.llm.offsite_tuning.emu_align.restore_from}.')
            else:
                assert adap_model is not None
                ckpt = torch.load(
                    cfg.llm.offsite_tuning.emu_align.restore_from,
                    map_location='cpu')
                adap_model.load_state_dict(ckpt['model'], strict=False)
                logger.info("Restored the adapter and emulator from ckpt")
                logger.warning(
                    "Please make sure the dtype of model keep the same.")
                # Make student un-trainable
                convert_layers_train_state(adap_model.student,
                                           is_trainable=False)
                does_train_emulator = False
        except Exception as error:
            logger.error(error)

    # Case1: Load ckpt, so we do not need to train student
    if not does_train_emulator:
        return adap_model

    # Case2: Restore fail or not assigned, start to train student
    new_cfg = build_cfg_for_alignment(cfg)

    # Make adapter un-trainable
    convert_layers_train_state(adap_model.adapter, is_trainable=False)

    # Make student trainable
    convert_layers_train_state(
        adap_model.student,
        name_pattern=adap_model.trainable_param_name_pattern,
        is_trainable=True)

    # Loading held-out data
    logger.info('Loading held-out dataset for alignment...')
    data, modified_cfg = get_data(new_cfg.clone())
    new_cfg.merge_from_other_cfg(modified_cfg)

    # Create `KDTrainer` and train
    kd_trainer = KDTrainer(raw_model,
                           adap_model,
                           data[1],
                           device,
                           new_cfg,
                           only_for_eval=False,
                           monitor=monitor)
    logger.info('Start to align student model with teacher model...')
    kd_trainer.train()
    logger.info('Alignment finished!')

    # Save aligned model
    del adap_model.teacher
    if cfg.llm.offsite_tuning.emu_align.save_to != '':
        adap_model.save_model(cfg.llm.offsite_tuning.emu_align.save_to)

    # Make adapter trainable
    convert_layers_train_state(
        adap_model.adapter,
        name_pattern=adap_model.trainable_param_name_pattern,
        is_trainable=True)

    # Make student un-trainable
    convert_layers_train_state(adap_model.student, is_trainable=False)

    return adap_model


def wrap_offsite_tuning_for_eval(model, config, ckpt_path=None):
    logger.info('===============use offsite tuning===============')
    print('===============use offsite tuning===============')
    # We use offsite-tuning in this experiment
    # Use adapter model instead
    adap_model = generate_adap_model(model, config.llm.offsite_tuning)
    # # Load kd model if ckpt exits
    # if config.llm.offsite_tuning.emu_align.use and \
    #         config.llm.offsite_tuning.eval_type == 'emu':
    #     if config.llm.offsite_tuning.emu_align.restore_from != '':
    #         try:
    #             ckpt = torch.load(
    #                 config.llm.offsite_tuning.emu_align.restore_from,
    #                 map_location='cpu',
    #             )
    #             adap_model.load_state_dict(ckpt['model'], strict=False)
    #             logger.info("Restored the adapter and emulator from ckpt")
    #         except Exception as error:
    #             logger.warning(error)

    # Load ckpt for eval
    try:
        if ckpt_path is None:
            ckpt_path = config.federate.save_to
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # # Sanity check
        # print('key for the loading model:')
        # print(ckpt['model'].keys())
        # print('key for the adapter+emulator model:')
        # adap_model_state_dict = adap_model.state_dict(return_trainable=False)
        # print(adap_model_state_dict.keys())
        # for key, value in ckpt['model'].items():
        #     print(key, torch.equal(value, adap_model_state_dict[key]))
        # exit()
        if 'model' in ckpt and 'cur_round' in ckpt:
            adap_model.load_state_dict(ckpt['model'])
            logger.info(f"Load with the model of Round {ckpt['cur_round']}")
        else:
            adap_model.load_state_dict(ckpt)
    except Exception as error:
        logger.warning(f"{error}, will use raw model.")

    if config.llm.offsite_tuning.eval_type == 'emu':
        logger.info("Evaluating for emulator+adapter...")
        print("Evaluating for emulator+adapter...")
        model = adap_model
        if hasattr(model, 'teacher'):
            del model.teacher
    elif config.llm.offsite_tuning.eval_type == 'full':
        # Raw model load adapter from adapter_and_emulator
        logger.info("Evaluating for full+adapter...")
        print("Evaluating for full+adapter...")
        new_model_adapter_state_dict = model.adapter.state_dict()
        for key, value in zip(new_model_adapter_state_dict.keys(),
                              adap_model.adapter.state_dict().values()):
            new_model_adapter_state_dict[key] = value
        model.adapter.load_state_dict(new_model_adapter_state_dict,
                                      strict=False)
        del adap_model
    else:
        raise NotImplementedError(
            '`config.llm.offsite_tuning.eval_type` should be chosen from '
            '`["emu", "full"]`.')
    return model
