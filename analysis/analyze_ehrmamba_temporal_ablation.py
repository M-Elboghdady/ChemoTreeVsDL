#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, fbeta_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from config.constants import PROJECT_ROOT
from ts_model_training.dataset import TimeSeriesDataset
from ts_model_training.envmanager import EnvManager
from ts_model_training.extractor import ExtractorTrain
from ts_model_training.factory import build_batcher, build_model
from ts_model_training.utils import set_all_seeds


# ============================== CONFIGURATION ==============================
TASK = 'sepsis'  # 'sepsis' or 'nf'
FOLDS = [0, 1, 2, 3, 4]
CONDITIONS = ['original', 'no_time', 'uniform_time', 'shuffle_events', 'reverse_events', 'shuffle_timestamps']
SHUFFLE_SEEDS = [42, 43, 44, 45, 46]

INFER_BATCH_SIZE = 128
DEFAULT_THRESHOLD = 0.5
FEATURE_THRESHOLD = True
STATIC_THRESHOLD = 0
HID_DIM_DEMO = 64
AGG_INT = 24
SPLIT_SEED = None
FAST = False
DROP_MINUTES = False
# ==========================================================================


RESULTS_ROOT = Path(PROJECT_ROOT) / 'MIMIC_IV/saved_data/results'
ANALYSIS_ROOT = Path(PROJECT_ROOT) / 'MIMIC_IV/saved_data/analysis'

TASK_CFG = {
    'nf': {
        'cohort': 'mimic_cohort_NF_30_days',
        'days': 14,
        'pt_base': RESULTS_ROOT / 'mimic_all/time_series/pretrain/ehrmamba_lab/ehrmamba_lab_pretrain_continuous_512_matched_layer',
        'ft_base': RESULTS_ROOT / 'mimic_cohort_NF_30_days/time_series/finetune/ehrmamba_lab',
        'ft_prefix': 'ehrmamba_pretrained_bestepochs_fold{fold}',
        'fold_configs': {
            0: PROJECT_DIR / 'config/ts_config_params.yaml',
            1: PROJECT_DIR / 'config/ehrmamba_fold1.yaml',
            2: PROJECT_DIR / 'config/ehrmamba_fold2.yaml',
            3: PROJECT_DIR / 'config/ehrmamba_fold3.yaml',
            4: PROJECT_DIR / 'config/ehrmamba_fold4.yaml',
        },
        'pt_layout': 'nf',
        'out_dir': ANALYSIS_ROOT / 'ehrmamba_temporal_ablation',
    },
    'sepsis': {
        'cohort': 'mimic_cohort_cancer_sepsis_icu24h',
        'days': 1,
        'pt_base': RESULTS_ROOT / 'mimic_all/time_series/pretrain/ehrmamba_lab/ehrmamba_lab_pretrain_excl_sepsis',
        'ft_base': RESULTS_ROOT / 'mimic_cohort_cancer_sepsis_icu24h/time_series/finetune/ehrmamba_lab',
        'ft_prefix': 'ehrmamba_cancer_sepsis_excl_sepsis_pt_fold{fold}',
        'fold_configs': {i: PROJECT_DIR / f'config/ehrmamba_fold{i}.yaml' for i in range(5)},
        'pt_layout': 'sepsis',
        'out_dir': ANALYSIS_ROOT / 'ehrmamba_temporal_ablation_cancer_sepsis',
    },
}

PAD_ID, CLS_ID, REG_ID, UNK_ID, LAB_ID_OFFSET = 0, 1, 2, 3, 4
ALL_CONDITIONS = ('original', 'no_time', 'uniform_time', 'shuffle_events', 'reverse_events', 'shuffle_timestamps')
RANDOM_CONDITIONS = {'shuffle_events', 'shuffle_timestamps'}


def cfg():
    if TASK not in TASK_CFG:
        raise ValueError(f'TASK must be one of {list(TASK_CFG)}, got {TASK!r}')
    invalid = set(CONDITIONS) - set(ALL_CONDITIONS)
    if invalid:
        raise ValueError(f'Unknown temporal conditions: {sorted(invalid)}')
    return TASK_CFG[TASK]


def make_cli():
    c = cfg()
    a = argparse.Namespace()
    a.dataset = 'MIMIC_IV'
    a.cohort = c['cohort']
    a.days_before_discharge = c['days']
    a.folds = list(FOLDS)
    a.pt_base = c['pt_base']
    a.ft_base = c['ft_base']
    a.ft_prefix_pattern = c['ft_prefix']
    a.out_dir = c['out_dir']
    a.conditions = list(CONDITIONS)
    a.shuffle_seeds = list(SHUFFLE_SEEDS)
    a.infer_batch_size = INFER_BATCH_SIZE
    a.threshold = DEFAULT_THRESHOLD
    a.feature_threshold = FEATURE_THRESHOLD
    a.static_threshold = STATIC_THRESHOLD
    a.hid_dim_demo = HID_DIM_DEMO
    a.agg_int = AGG_INT
    a.split_seed = SPLIT_SEED
    a.fast = FAST
    a.drop_minutes = DROP_MINUTES
    if not a.shuffle_seeds:
        raise ValueError('SHUFFLE_SEEDS cannot be empty')
    return a


def config_for_fold(fold: int) -> Path:
    configs = cfg()['fold_configs']
    if fold not in configs:
        raise KeyError(f'No config for fold {fold}')
    return configs[fold]


def pretrain_dir_for_fold(args, fold: int) -> Path:
    if cfg()['pt_layout'] == 'sepsis':
        return args.pt_base / f'fold{fold}' / 'fold_0' / 'grid_none'
    if fold == 0:
        return args.pt_base / 'fold_0' / 'grid_none'
    return args.pt_base / f'nf_fold{fold}' / 'fold_0' / 'grid_none'


def finetune_dir_for_fold(args, fold: int) -> Path:
    return args.ft_base / args.ft_prefix_pattern.format(fold=fold) / f'fold_{fold}' / 'grid_best_epochs'


def require_file(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f'{what} not found: {path}')


def make_pipeline_args(cli, fold: int, config_path: Path, pt_dir: Path):
    a = argparse.Namespace()
    a.days_before_discharge = cli.days_before_discharge
    a.dataset = cli.dataset
    a.cohort = cli.cohort
    a.fold = fold
    a.model_type = 'ehrmamba_lab'
    a.feature_threshold = bool(cli.feature_threshold)
    a.static_threshold = cli.static_threshold
    a.hid_dim_demo = cli.hid_dim_demo
    a.agg_int = cli.agg_int
    a.prefix = f'ehrmamba_temporal_ablation_runtime_fold{fold}'
    a.config_path = str(config_path)
    a.split_seed = cli.split_seed
    a.fast = bool(cli.fast)
    a.grid = 'none'
    a.pretrain = False
    a.load_ckpt_path = str(pt_dir)
    a.drop_minutes = bool(cli.drop_minutes)
    a.freeze = False
    a.infer_batch_size = cli.infer_batch_size
    a.ml_features = ['LAB', 'DEMO']
    a.ml_feature_method = 'concatenate'
    return a


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _sample_seed(base_seed: int, sample_index: int) -> int:
    return int((int(base_seed) * 1000003 + int(sample_index)) % (2 ** 63 - 1))


def _randperm_cpu(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return torch.randperm(n, generator=generator)


def perturb_temporal_batch(
    batch: Dict[str, torch.Tensor], condition: str = 'original',
    sample_indices: Optional[Sequence[int]] = None, seed: int = 42,
) -> Dict[str, torch.Tensor]:
    if condition not in ALL_CONDITIONS:
        raise ValueError(f'Unknown temporal condition: {condition}')
    if condition == 'original':
        return batch

    required = {'event_ids', 'values', 'event_times', 'lengths'}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f'EHRMamba batch is missing required keys: {sorted(missing)}')

    out = _clone_batch(batch)
    event_ids, values = out['event_ids'], out['values']
    event_times, lengths = out['event_times'], out['lengths']

    if event_ids.ndim != 2 or values.shape != event_ids.shape or event_times.shape != event_ids.shape:
        raise ValueError('Expected event_ids, values, event_times with identical [B, L] shapes')

    batch_size, max_len = event_ids.shape
    if sample_indices is None:
        sample_indices = list(range(batch_size))
    if len(sample_indices) != batch_size:
        raise ValueError('sample_indices length does not match batch size')

    for b in range(batch_size):
        length = max(0, min(int(lengths[b].item()), max_len))
        if length == 0:
            continue

        positions = torch.arange(length, device=event_ids.device)
        real_idx = positions[event_ids[b, :length] >= LAB_ID_OFFSET]
        n = int(real_idx.numel())
        if n <= 1:
            continue

        if condition == 'no_time':
            event_times[b, real_idx] = 0

        elif condition == 'uniform_time':
            old_times = event_times[b, real_idx].clone()
            t_min, t_max = old_times.min(), old_times.max()
            if float((t_max - t_min).item()) > 0:
                uniform = torch.linspace(
                    float(t_min.item()), float(t_max.item()), steps=n,
                    device=event_times.device, dtype=torch.float32,
                ).to(dtype=event_times.dtype)
                event_times[b, real_idx] = uniform

        elif condition == 'shuffle_events':
            perm = _randperm_cpu(n, _sample_seed(seed, int(sample_indices[b]))).to(real_idx.device)
            old_ids, old_values = event_ids[b, real_idx].clone(), values[b, real_idx].clone()
            event_ids[b, real_idx], values[b, real_idx] = old_ids[perm], old_values[perm]

        elif condition == 'reverse_events':
            old_ids, old_values = event_ids[b, real_idx].clone(), values[b, real_idx].clone()
            event_ids[b, real_idx] = torch.flip(old_ids, dims=[0])
            values[b, real_idx] = torch.flip(old_values, dims=[0])

        elif condition == 'shuffle_timestamps':
            perm = _randperm_cpu(n, _sample_seed(seed, int(sample_indices[b]))).to(real_idx.device)
            old_times = event_times[b, real_idx].clone()
            event_times[b, real_idx] = old_times[perm]

    return out


@torch.no_grad()
def predict_indices(
    model, batcher, indices: np.ndarray, device, batch_size: int,
    temporal_condition: str = 'original', temporal_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, logits_all = [], []
    indices = np.asarray(indices, dtype=int)

    for start in range(0, len(indices), batch_size):
        ind = indices[start:start + batch_size]
        batch = batcher.get_batch(ind)
        batch = perturb_temporal_batch(batch, temporal_condition, ind, temporal_seed)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(**batch)
        logits = logits.detach()
        logits_all.append(logits.cpu().numpy().reshape(-1))
        probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))

    return np.concatenate(probs), np.concatenate(logits_all)


def seeds_for_condition(condition: str, seeds: Sequence[int]) -> List[Optional[int]]:
    return [int(seed) for seed in seeds] if condition in RANDOM_CONDITIONS else [None]


def infer_one_fold(cli, fold: int) -> pd.DataFrame:
    config_path = config_for_fold(fold)
    pt_dir = pretrain_dir_for_fold(cli, fold)
    ft_dir = finetune_dir_for_fold(cli, fold)
    ft_ckpt = ft_dir / 'checkpoint_best.bin'

    require_file(config_path, f'fold {fold} config')
    require_file(pt_dir / 'pt_saved_variables.pkl', f'fold {fold} PT metadata')
    require_file(pt_dir / 'checkpoint_best.bin', f'fold {fold} PT checkpoint')
    require_file(ft_ckpt, f'fold {fold} final fine-tuned checkpoint')

    args = make_pipeline_args(cli, fold, config_path, pt_dir)
    env = EnvManager(args)
    ExtractorTrain(env.args)
    env.set_model_params(mode='default')
    env.set_ids(mode='default')
    set_all_seeds(env.args.seed, env.args.fast)
    env.args.cv_mode = 'grid'

    dataset = TimeSeriesDataset(env.args)
    model = build_model(env.args)
    batcher = build_batcher(env.args, dataset.preproc.input_dict)
    model.load_state_dict(torch.load(ft_ckpt, map_location=env.args.device), strict=True)
    model.to(env.args.device).eval()

    test_ind = np.asarray(batcher.splits['test'], dtype=int)
    if len(test_ind) == 0:
        raise RuntimeError(f'Fold {fold}: test split is empty')
    if 'sup_ids' not in env.args.ids:
        raise KeyError("args.ids['sup_ids'] missing after TimeSeriesDataset construction")

    hadm = np.asarray(env.args.ids['sup_ids'])[test_ind].astype(np.int64)
    y = np.asarray(batcher.y).reshape(-1)[test_ind].astype(int)
    rows = []

    for condition in cli.conditions:
        for temporal_seed in seeds_for_condition(condition, cli.shuffle_seeds):
            seed = 42 if temporal_seed is None else int(temporal_seed)
            print(f'  condition={condition}' + ('' if temporal_seed is None else f' seed={temporal_seed}'))
            prob, logit = predict_indices(
                model, batcher, test_ind, env.args.device, cli.infer_batch_size,
                temporal_condition=condition, temporal_seed=seed,
            )
            rows.append(pd.DataFrame({
                'hadm_id': hadm, 'fold': fold, 'label': y, 'condition': condition,
                'seed': np.nan if temporal_seed is None else temporal_seed,
                'probability': prob, 'logit': logit,
            }))

    del model, batcher, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True)


def min_recall_precision(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    return np.nan if len(precision) == 0 else float(np.max(np.minimum(precision, recall)))


def metrics_from_arrays(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    n, n_pos = len(y), int(y.sum())

    out = {
        'n': n, 'n_pos': n_pos, 'prevalence': float(y.mean()) if n else np.nan,
        'mean_prob': float(p.mean()) if n else np.nan,
        'brier': float(brier_score_loss(y, p)) if n else np.nan,
        'precision': float(precision_score(y, pred, zero_division=0)) if n else np.nan,
        'recall': float(recall_score(y, pred, zero_division=0)) if n else np.nan,
        'f1': float(f1_score(y, pred, zero_division=0)) if n else np.nan,
        'f2': float(fbeta_score(y, pred, beta=2, zero_division=0)) if n else np.nan,
        'balanced_acc': float(balanced_accuracy_score(y, pred)) if n else np.nan,
        'mcc': float(matthews_corrcoef(y, pred)) if n else np.nan,
    }

    if 0 < n_pos < n:
        out['auroc'] = float(roc_auc_score(y, p))
        out['auprc'] = float(average_precision_score(y, p))
        out['minrp'] = min_recall_precision(y, p)
    else:
        out['auroc'] = out['auprc'] = out['minrp'] = np.nan

    return out


def metric_table(pred: pd.DataFrame, threshold: float, by_fold: bool) -> pd.DataFrame:
    group_cols = ['fold', 'condition', 'seed'] if by_fold else ['condition', 'seed']
    rows = []

    for key, group in pred.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))
        rec.update(metrics_from_arrays(group['label'].values, group['probability'].values, threshold))
        rows.append(rec)

    return pd.DataFrame(rows)


def add_deltas_vs_original(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    metric_cols = ['auroc', 'auprc', 'brier', 'minrp']

    if 'fold' in out.columns:
        originals = out[out['condition'] == 'original'].set_index('fold')[metric_cols]
        originals = originals.rename(columns={c: f'original_{c}' for c in metric_cols})
        out = out.join(originals, on='fold')
    else:
        original_rows = out[out['condition'] == 'original']
        for c in metric_cols:
            out[f'original_{c}'] = original_rows.iloc[0][c] if len(original_rows) == 1 else np.nan

    for c in metric_cols:
        out[f'delta_{c}_vs_original'] = out[c] - out[f'original_{c}']
    return out


def add_prediction_shift(pred: pd.DataFrame) -> pd.DataFrame:
    original = pred[pred['condition'] == 'original'][['hadm_id', 'fold', 'probability']]
    original = original.drop_duplicates(['hadm_id', 'fold']).rename(columns={'probability': 'original_probability'})

    if original.empty:
        pred = pred.copy()
        pred['original_probability'] = np.nan
        pred['delta_probability_vs_original'] = np.nan
        pred['abs_delta_probability_vs_original'] = np.nan
        return pred

    out = pred.merge(original, on=['hadm_id', 'fold'], how='left', validate='many_to_one')
    out['delta_probability_vs_original'] = out['probability'] - out['original_probability']
    out['abs_delta_probability_vs_original'] = out['delta_probability_vs_original'].abs()
    return out


def prediction_shift_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, seed), group in pred.groupby(['condition', 'seed'], dropna=False, sort=False):
        x = group['abs_delta_probability_vs_original'].dropna().to_numpy(float)
        d = group['delta_probability_vs_original'].dropna().to_numpy(float)
        rows.append({
            'condition': condition, 'seed': seed, 'n': len(x),
            'mean_abs_prob_shift': float(np.mean(x)) if len(x) else np.nan,
            'median_abs_prob_shift': float(np.median(x)) if len(x) else np.nan,
            'p90_abs_prob_shift': float(np.percentile(x, 90)) if len(x) else np.nan,
            'mean_signed_prob_shift': float(np.mean(d)) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'auroc', 'auprc', 'brier', 'minrp', 'delta_auroc_vs_original', 'delta_auprc_vs_original',
        'delta_brier_vs_original', 'delta_minrp_vs_original',
    ]
    rows = []

    for condition, group in metrics.groupby('condition', sort=False):
        rec = {'condition': condition, 'n_runs': len(group)}
        for c in cols:
            values = pd.to_numeric(group[c], errors='coerce')
            rec[f'{c}_mean'] = float(values.mean()) if values.notna().any() else np.nan
            rec[f'{c}_std'] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(rec)

    return pd.DataFrame(rows)


def fold_averaged_summary(by_fold: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        'auroc', 'auprc', 'brier', 'minrp', 'delta_auroc_vs_original', 'delta_auprc_vs_original',
        'delta_brier_vs_original', 'delta_minrp_vs_original',
    ]

    per_fold = by_fold.groupby(['fold', 'condition'], as_index=False)[cols].mean()
    rows = []

    for condition, group in per_fold.groupby('condition', sort=False):
        rec = {'condition': condition, 'n_folds': int(group['fold'].nunique())}
        for c in cols:
            values = pd.to_numeric(group[c], errors='coerce')
            rec[f'{c}_mean'] = float(values.mean()) if values.notna().any() else np.nan
            rec[f'{c}_std'] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(rec)

    return per_fold, pd.DataFrame(rows)


def main():
    cli = make_cli()
    cli.out_dir.mkdir(parents=True, exist_ok=True)

    if 'original' not in cli.conditions:
        print("WARNING: 'original' not requested; delta-vs-original columns will be NaN.")

    print(f'TASK={TASK} | cohort={cli.cohort}')
    print(f"Conditions: {', '.join(cli.conditions)}")
    print(f'Shuffle seeds: {cli.shuffle_seeds}')

    all_predictions = []
    for fold in cli.folds:
        print(f'\n===== FOLD {fold} =====')
        all_predictions.append(infer_one_fold(cli, fold))

    pred = pd.concat(all_predictions, ignore_index=True)
    duplicate_cols = ['hadm_id', 'fold', 'condition', 'seed']
    if pred.duplicated(duplicate_cols).any():
        raise RuntimeError('Duplicate admission/condition predictions detected')

    pred = add_prediction_shift(pred)
    overall = add_deltas_vs_original(metric_table(pred, cli.threshold, by_fold=False))
    by_fold = add_deltas_vs_original(metric_table(pred, cli.threshold, by_fold=True))
    shifts = prediction_shift_summary(pred)
    seeds = seed_summary(overall)
    per_fold, fold_summary = fold_averaged_summary(by_fold)

    paths = {
        'pred': cli.out_dir / 'oof_temporal_predictions.csv',
        'overall': cli.out_dir / 'temporal_metrics_overall.csv',
        'fold': cli.out_dir / 'temporal_metrics_by_fold.csv',
        'seed': cli.out_dir / 'temporal_metrics_seed_summary.csv',
        'shift': cli.out_dir / 'temporal_probability_shift_summary.csv',
        'per_fold': cli.out_dir / 'temporal_seed_averaged_per_fold.csv',
        'fold_summary': cli.out_dir / 'temporal_fold_averaged_summary.csv',
    }

    for key, dataframe in [
        ('pred', pred), ('overall', overall), ('fold', by_fold), ('seed', seeds),
        ('shift', shifts), ('per_fold', per_fold), ('fold_summary', fold_summary),
    ]:
        dataframe.to_csv(paths[key], index=False)

    display_cols = [
        'condition', 'auroc_mean', 'auroc_std', 'delta_auroc_vs_original_mean',
        'auprc_mean', 'auprc_std', 'delta_auprc_vs_original_mean',
    ]

    print('\n===== FOLD-AVERAGED PRIMARY RESULTS =====')
    print(fold_summary[display_cols].to_string(index=False))
    print('\n===== DONE =====')
    for key, path in paths.items():
        print(f'{key:12s}: {path}')


if __name__ == '__main__':
    main()

