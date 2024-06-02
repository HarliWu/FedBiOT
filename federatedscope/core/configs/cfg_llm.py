import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # LLM related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN(new_allowed=True)
    cfg.llm.tok_len = 128
    cfg.llm.max_new_token = 60
    cfg.llm.num_completions = 2
    cfg.llm.retry_on_nan_loss = False

    # Training the reward model
    cfg.llm.reward_coeff = 0.1

    # ---------------------------------------------------------------------- #
    # RLHF for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.rlhf = False

    # Enable alternative training (reward -> policy -> reward -> ...)
    cfg.llm.fedrlhf = CN()
    cfg.llm.fedrlhf.use = False
    cfg.llm.fedrlhf.config_file = ''
    cfg.llm.fedrlhf.pretrained = False
    cfg.llm.fedrlhf.frequency = 100
    cfg.llm.fedrlhf.train = CN()
    cfg.llm.fedrlhf.train.local_update_steps = 10
    cfg.llm.fedrlhf.train.batch_or_epoch = 'batch'

    # use gradient accumulation to enlarge the batch size
    # e.g., bsz = dataloader.batch_size * train.grad_accu_step
    cfg.llm.grad_accum_step = 1

    # use gradient accumulation to enlarge the batch size
    # e.g., bsz = dataloader.batch_size * train.grad_accu_step
    cfg.llm.grad_accum_step = 1

    # ---------------------------------------------------------------------- #
    # Cache for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.cache = CN()
    cfg.llm.cache.model = ''

    # ---------------------------------------------------------------------- #
    # Chat tools for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.chat = CN()
    cfg.llm.chat.max_history_len = 10
    cfg.llm.chat.max_len = 100

    # ---------------------------------------------------------------------- #
    # Deepspeed related options
    # ---------------------------------------------------------------------- #
    cfg.llm.deepspeed = CN()
    cfg.llm.deepspeed.use = False
    cfg.llm.deepspeed.ds_config = ''

    # ---------------------------------------------------------------------- #
    # HuggingFace accelerator related options
    # ---------------------------------------------------------------------- #
    cfg.llm.accelerator = CN()
    cfg.llm.accelerator.use = False
    cfg.llm.accelerator.config = ''

    # ---------------------------------------------------------------------- #
    # Adapters for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter = CN()
    cfg.llm.adapter.use = False
    cfg.llm.adapter.count = 1
    cfg.llm.adapter.warmup = CN()
    cfg.llm.adapter.warmup.use = False
    cfg.llm.adapter.warmup.round = 10  # warmup round for each adapter
    cfg.llm.adapter.args = [{}]
    # Move adapter to `cpu` after training, which can save memory but cost
    # more time.
    cfg.llm.adapter.mv_to_cpu = False

    # Each client trains their own local model for different adapters
    cfg.llm.adapter.local_only = False

    # Cluster-wise method to ensure every adapter carries the number of clients
    cfg.llm.adapter.grouping = CN()
    cfg.llm.adapter.grouping.use = False
    cfg.llm.adapter.grouping.round = 50

    # ---------------------------------------------------------------------- #
    # Offsite-tuning related options
    # ---------------------------------------------------------------------- #
    cfg.llm.offsite_tuning = CN()
    cfg.llm.offsite_tuning.use = False
    cfg.llm.offsite_tuning.strategy = 'drop_layer'
    cfg.llm.offsite_tuning.kwargs = [{}]
    cfg.llm.offsite_tuning.emu_l = 1  # Index of emulator layer left
    cfg.llm.offsite_tuning.emu_r = 10  # Index of emulator layer right

    # Used in `eval`
    cfg.llm.offsite_tuning.eval_type = 'emu'  # Choose one of `[emu, full]`

    # Used in `aggregator`
    cfg.llm.offsite_tuning.save_full_model = False

    # Emulator alignment will use dataset in Server
    cfg.llm.offsite_tuning.emu_align = CN()
    cfg.llm.offsite_tuning.emu_align.use = False
    cfg.llm.offsite_tuning.emu_align.initial_only = True
    cfg.llm.offsite_tuning.emu_align.init_enable_ground_truth = False
    cfg.llm.offsite_tuning.emu_align.sim_loss = 'l2'  # Choose one of
    # `['cos', 'l2']`
    cfg.llm.offsite_tuning.emu_align.layerwise_distill = False
    cfg.llm.offsite_tuning.emu_align.kl_divergence = 'raw'  # Choose one of
    # `['raw', 'logps']`
    cfg.llm.offsite_tuning.emu_align.restore_from = ''
    cfg.llm.offsite_tuning.emu_align.save_to = ''
    cfg.llm.offsite_tuning.emu_align.exit_after_align = False

    # Server held-out data
    cfg.llm.offsite_tuning.emu_align.data = CN()
    cfg.llm.offsite_tuning.emu_align.data.root = 'data'
    cfg.llm.offsite_tuning.emu_align.data.type = 'alpaca@llm'
    cfg.llm.offsite_tuning.emu_align.data.splits = [0.8, 0.1, 0.1]

    cfg.llm.offsite_tuning.emu_align.train = CN()
    cfg.llm.offsite_tuning.emu_align.train.local_update_steps = 10
    cfg.llm.offsite_tuning.emu_align.train.initial_update_rounds = 50
    cfg.llm.offsite_tuning.emu_align.train.batch_or_epoch = 'batch'
    cfg.llm.offsite_tuning.emu_align.train.lm_loss_weight = 0.1
    cfg.llm.offsite_tuning.emu_align.train.kd_loss_weight = 0.9
    cfg.llm.offsite_tuning.emu_align.train.enable_ground_truth = False

    cfg.llm.offsite_tuning.emu_align.train.optimizer = CN(new_allowed=True)
    cfg.llm.offsite_tuning.emu_align.train.optimizer.type = 'SGD'
    cfg.llm.offsite_tuning.emu_align.train.optimizer.lr = 0.01

    # Overwrite clients' labels to LLM generated text
    cfg.llm.offsite_tuning.llm_generated = CN()
    cfg.llm.offsite_tuning.llm_generated.use = False
    cfg.llm.offsite_tuning.llm_generated.ratio = 0.1


def assert_llm_cfg(cfg):
    if cfg.llm.offsite_tuning.emu_align.use:
        if cfg.llm.offsite_tuning.emu_align.restore_from != '':
            logger.warning(
                'Enabling `restore_from` in offsite_tuning emulator '
                'alignment will skip training the emulator.')


register_config("llm", extend_llm_cfg)
