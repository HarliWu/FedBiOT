# def generate_unnamed_adapter(model: AdapterModel,
#                              split_idx=1000,
#                              adapter_size=1,
#                              **kwargs):
#     layers = get_layers(model)
#     adapter_size = int(max(adapter_size, 1))
#     cutter = min(split_idx, len(layers) - adapter_size)

#     # Freeze the entire model
#     for layer in layers:
#         for param in layer.parameters():
#             param.data = param.data.float()
#             param.requires_grad = False

#     # Unfreeze the selected layers as the adapter
#     # Generate the adapter using uniform layer dropout
#     adapter = nn.ModuleList()
#     for a in np.array_split(np.arange(cutter, len(layers)), adapter_size):
#         for param in layers[a[-1]].parameters():
#             param.requires_grad = True
#         logger.info(f"Adding layer {a[-1]} to adapter.")
#         adapter.append(layers[a[-1]])

#     new_model_layers = nn.ModuleList()
#     for idx in range(cutter):
#         new_model_layers.append(layers[idx])

#     for idx in range(len(adapter)):
#         new_model_layers.append(adapter[idx])

#     new_model = copy.deepcopy(model)
#     new_model = set_layers(new_model, new_model_layers)

#     gc.collect()
#     torch.cuda.empty_cache()

#     return new_model

# def generate_adap_model(model: AdapterModel, offsite_tuning_cfg):
#     if offsite_tuning_cfg.strategy in COMP_FUNC_MAPPING.keys():
#         compress_strategy = offsite_tuning_cfg.strategy
#         emulator_l = offsite_tuning_cfg.emu_l
#         emulator_r = offsite_tuning_cfg.emu_r
#         emu_align = offsite_tuning_cfg.emu_align.use
#         offsite_tuning_kwargs = offsite_tuning_cfg.kwargs[0]
#         return generate_emulator_and_adapter(
#             model,
#             strategy=compress_strategy,
#             emulator_l=emulator_l,
#             emulator_r=emulator_r,
#             emulator_alignment=emu_align,
#             **offsite_tuning_kwargs
#         )
#     else:
#         # split_idx = offsite_tuning_cfg.emu_r
#         # offsite_tuning_kwargs = offsite_tuning_cfg.kwargs[0]
#         # return generate_unnamed_adapter(model, split_idx,
#         #                                 **offsite_tuning_kwargs)
#         raise NotImplementedError
