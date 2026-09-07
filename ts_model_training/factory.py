from ts_model_training.ts_models.ts_gru import GRU_TS
from ts_model_training.ts_models.ts_sand import SAND_TS
from ts_model_training.ts_models.ts_tcn import TCN_TS
from ts_model_training.ts_models.ts_lstm import LSTM_TS
from ts_model_training.ts_models.ts_grud import GRUD_TS
from ts_model_training.ts_models.ts_interpnet import INTERPNET_TS
from ts_model_training.ts_models.ts_strats import STRATS_TS
from ts_model_training.ts_models.mlp import MLP
from ts_model_training.ts_models.ts_mamba import MAMBA_TS
from ts_model_training.ts_models.ts_mamba_fusion import MAMBA_FUSION_TS
from ts_model_training.batcher import Batcher, BatcherA, BatcherB, BatcherC_sup, BatcherC_unsup

MODEL_CLASSES = {
    'gru': GRU_TS,
    'tcn': TCN_TS,
    'lstm': LSTM_TS,
    'sand': SAND_TS,
    'grud': GRUD_TS,
    'interpnet': INTERPNET_TS,
    'strats': STRATS_TS,
    'mlp': MLP,
    "mamba": MAMBA_TS,
    "mamba_fusion": MAMBA_FUSION_TS,
}

def build_model(args):
    model_cls = MODEL_CLASSES.get(args.model_type.lower())
    if model_cls is None:
        raise ValueError(
            f"Unknown model type '{args.model_type}'. "
            f"Available: {list(MODEL_CLASSES)}"
        )
    model = model_cls(args)
    return model.to(args.device)

def build_batcher(args, input_dict):
    # get batcher based on model
    model_type = args.model_type
    if model_type in ['gru', 'lstm', 'tcn', 'sand', 'mlp', 'mamba', 'mamba_fusion']:
        batcher = BatcherA(args, input_dict)
    elif model_type in ['grud', 'interpnet']:
        batcher = BatcherB(args, input_dict)
    elif model_type in ['strats', 'istrats'] and args.train_mode == "pretrain":
        batcher = BatcherC_unsup(args, input_dict)
    elif model_type in ['strats', 'istrats'] and args.train_mode != "pretrain":
        batcher = BatcherC_sup(args, input_dict)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    args.logger.write('\nBatching module assigned.')
    return batcher
