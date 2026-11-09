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
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, fbeta_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from config.constants import PROJECT_ROOT
from ts_model_training.dataset import TimeSeriesDataset
from ts_model_training.envmanager import EnvManager
from ts_model_training.extractor import ExtractorTrain
from ts_model_training.factory import build_batcher, build_model
from ts_model_training.utils import set_all_seeds
# ============================== CONFIGURATION ==============================
TASK = 'sepsis'                  # 'sepsis' or 'nf'
METHOD = 'stratified'            # 'stratified' or 'adaptive'
FOLDS = [0, 1, 2, 3, 4]
RETENTION_LEVELS = [1.0, 0.75, 0.5, 0.25]
SUBSAMPLE_SEEDS = [42, 43, 44, 45, 46]

SPARSE_LAB_THRESHOLD = 0
TIME_BINS = 4                    # stratified only
VALUE_BINS = 4                   # stratified only
PROTECT_TIME_ENDPOINTS = True

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
    },
    'sepsis': {
        'cohort': 'mimic_cohort_cancer_sepsis_icu24h',
        'days': 1,
        'pt_base': RESULTS_ROOT / 'mimic_all/time_series/pretrain/ehrmamba_lab/ehrmamba_lab_pretrain_excl_sepsis',
        'ft_base': RESULTS_ROOT / 'mimic_cohort_cancer_sepsis_icu24h/time_series/finetune/ehrmamba_lab',
        'ft_prefix': 'ehrmamba_cancer_sepsis_excl_sepsis_pt_fold{fold}',
        'fold_configs': {
            i: PROJECT_DIR / f'config/ehrmamba_fold{i}.yaml' for i in range(5)
        },
        'pt_layout': 'sepsis',
    },
}

PAD_ID, CLS_ID, REG_ID, UNK_ID, LAB_ID_OFFSET = (0, 1, 2, 3, 4)


def cfg():
    if TASK not in TASK_CFG:
        raise ValueError(f'TASK must be one of {list(TASK_CFG)}, got {TASK!r}')
    if METHOD not in {'stratified', 'adaptive'}:
        raise ValueError("METHOD must be 'stratified' or 'adaptive'")
    return TASK_CFG[TASK]


def analysis_name():
    base = 'event_subsampling' if METHOD == 'stratified' else 'adaptive_subsampling'
    return f'ehrmamba_{base}' + ('_cancer_sepsis' if TASK == 'sepsis' else '')


def make_cli():
    c = cfg()
    a = argparse.Namespace()
    a.dataset = 'MIMIC_IV'
    a.cohort = c['cohort']
    a.days_before_discharge = c['days']
    a.folds = FOLDS
    a.pt_base = c['pt_base']
    a.ft_base = c['ft_base']
    a.ft_prefix_pattern = c['ft_prefix']
    a.out_dir = ANALYSIS_ROOT / analysis_name()
    a.retention_levels = sorted({float(x) for x in RETENTION_LEVELS}, reverse=True)
    a.subsample_seeds = list(SUBSAMPLE_SEEDS)
    a.sparse_lab_threshold = SPARSE_LAB_THRESHOLD
    a.time_bins = TIME_BINS
    a.value_bins = VALUE_BINS
    a.protect_time_endpoints = PROTECT_TIME_ENDPOINTS
    a.infer_batch_size = INFER_BATCH_SIZE
    a.threshold = DEFAULT_THRESHOLD
    a.feature_threshold = FEATURE_THRESHOLD
    a.static_threshold = STATIC_THRESHOLD
    a.hid_dim_demo = HID_DIM_DEMO
    a.agg_int = AGG_INT
    a.split_seed = SPLIT_SEED
    a.fast = FAST
    a.drop_minutes = DROP_MINUTES
    if not any((np.isclose(x, 1.0) for x in a.retention_levels)):
        a.retention_levels = [1.0] + a.retention_levels
    if any((x <= 0 or x > 1 for x in a.retention_levels)):
        raise ValueError('RETENTION_LEVELS must be in (0,1]')
    if not a.subsample_seeds:
        raise ValueError('SUBSAMPLE_SEEDS cannot be empty')
    if a.sparse_lab_threshold < 0:
        raise ValueError('SPARSE_LAB_THRESHOLD must be >= 0')
    if METHOD == 'stratified' and (a.time_bins < 1 or a.value_bins < 1):
        raise ValueError('TIME_BINS and VALUE_BINS must be >= 1')
    return a


def config_for_fold(fold: int) -> Path:
    m = cfg()['fold_configs']
    if fold not in m:
        raise KeyError(f'No config for fold {fold}')
    return m[fold]


def pretrain_dir_for_fold(args, fold: int) -> Path:
    if cfg()['pt_layout'] == 'sepsis':
        return args.pt_base / f'fold{fold}' / 'fold_0' / 'grid_none'
    if fold == 0:
        return args.pt_base / 'fold_0/grid_none'
    return args.pt_base / f'nf_fold{fold}/fold_0/grid_none'


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
    a.prefix = f'ehrmamba_{METHOD}_subsampling_runtime_fold{fold}'
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


def _clone_batch(batch):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _mix_seed(seed: int, sample: int, lab: int, retention: float) -> int:
    r = int(round(float(retention) * 10000))
    mixed = int(seed) * 1000003 + int(sample) * 9176 + int(lab) * 6113 + r * 3571
    return int(mixed % (2 ** 63 - 1))


def _equal_width_bins(x, n):
    x = np.asarray(x, float)
    if len(x) == 0 or n <= 1 or (not np.isfinite(x).all()) or (np.ptp(x) == 0):
        return np.zeros(len(x), int)
    e = np.linspace(float(x.min()), float(x.max()), int(n) + 1)
    return np.clip(np.searchsorted(e[1:-1], x, side='right'), 0, int(n) - 1).astype(int)


def _quantile_bins(x, n):
    x = np.asarray(x, float)
    if len(x) == 0 or n <= 1 or (not np.isfinite(x).all()) or (np.ptp(x) == 0):
        return np.zeros(len(x), int)
    e = np.unique(np.quantile(x, np.linspace(0, 1, int(n) + 1)))
    if len(e) <= 2:
        return np.zeros(len(x), int)
    return np.clip(np.searchsorted(e[1:-1], x, side='right'), 0, len(e) - 2).astype(int)


def _allocate_proportionally(strata, target, rng):
    labels, counts = np.unique(strata, return_counts=True)
    total = int(counts.sum())
    if target <= 0 or total == 0:
        return {int(k): 0 for k in labels}
    if target >= total:
        return {int(k): int(c) for k, c in zip(labels, counts)}
    raw = target * counts.astype(float) / total
    alloc = np.floor(raw).astype(int)
    rem = int(target - alloc.sum())
    if rem > 0:
        order = np.lexsort((rng.random(len(labels)), -(raw - alloc)))
        for j in order[:rem]:
            alloc[j] += 1
    return {int(k): int(v) for k, v in zip(labels, alloc)}


def _stratified_select(pos, times, vals, keep_n, rng):
    n = len(pos)
    if keep_n >= n:
        return np.sort(pos.copy())
    if keep_n <= 0:
        return np.empty(0, int)
    order = np.argsort(times, kind='stable')
    pos, times, vals = (pos[order], times[order], vals[order])
    if PROTECT_TIME_ENDPOINTS and keep_n >= 2 and n >= 2:
        chosen = [0, n - 1]
    elif PROTECT_TIME_ENDPOINTS and keep_n == 1:
        chosen = [int(rng.integers(0, n))]
    else:
        chosen = []
    need = keep_n - len(chosen)
    if need > 0:
        all_i = np.arange(n)
        mask = np.ones(n, bool)
        if chosen:
            mask[np.asarray(chosen)] = False
        cand = all_i[mask]
        strata = (
            _equal_width_bins(times[cand], TIME_BINS) * max(1, VALUE_BINS)
            + _quantile_bins(vals[cand], VALUE_BINS)
        )
        picked = []
        for s, k in _allocate_proportionally(strata, need, rng).items():
            if k <= 0:
                continue
            m = cand[strata == s]
            if k >= len(m):
                picked += m.tolist()
            else:
                picked += rng.choice(m, size=k, replace=False).tolist()
        chosen += picked
    if len(chosen) < keep_n:
        rem = np.setdiff1d(np.arange(n), np.asarray(chosen, int), assume_unique=False)
        chosen += rng.choice(rem, size=keep_n - len(chosen), replace=False).tolist()
    chosen = np.asarray(sorted(set(chosen)), int)[:keep_n]
    return np.sort(pos[chosen])


def _local_change_importance(values):
    v = np.nan_to_num(np.asarray(values, float), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(v)
    if n == 0:
        return np.empty(0, float)
    if n == 1:
        return np.ones(1, float)
    s = np.zeros(n, float)
    s[0] = abs(v[1] - v[0])
    s[-1] = abs(v[-1] - v[-2])
    if n > 2:
        s[1:-1] = np.abs(v[1:-1] - v[:-2]) + np.abs(v[2:] - v[1:-1])
    return s


def _adaptive_select(pos, times, vals, keep_n, rng):
    n = len(pos)
    order = np.argsort(times, kind='stable')
    pos, vals = (pos[order], vals[order])
    scores = _local_change_importance(vals)
    if keep_n >= n:
        return (np.sort(pos.copy()), scores)
    if keep_n <= 0:
        return (np.empty(0, int), scores)
    if PROTECT_TIME_ENDPOINTS and n >= 2 and keep_n >= 2:
        protected = [0, n - 1]
    elif PROTECT_TIME_ENDPOINTS and keep_n == 1:
        protected = [0 if scores[0] >= scores[-1] else n - 1]
    else:
        protected = []
    selected = set(protected)
    need = keep_n - len(selected)
    if need > 0:
        cand = np.asarray([i for i in range(n) if i not in selected], int)
        rank = np.lexsort((rng.random(len(cand)), -scores[cand]))
        selected.update(map(int, cand[rank[:need]]))
    return (np.sort(pos[np.asarray(sorted(selected), int)]), scores)


def _select_patient_events(lab_ids, vals, times, positions, retention, seed, sample_idx, sparse_threshold):
    selected, score_map = ([], {})
    for lab in np.unique(lab_ids):
        m = lab_ids == lab
        pos, v, t = (positions[m], vals[m], times[m])
        n = len(pos)
        keep_n = n if retention >= 1 or n <= sparse_threshold else max(1, int(round(retention * n)))
        rng = np.random.default_rng(_mix_seed(seed, sample_idx, int(lab), retention))
        if METHOD == 'stratified':
            chosen = _stratified_select(pos, t, v, keep_n, rng)
        else:
            chosen, scores = _adaptive_select(pos, t, v, keep_n, rng)
            order = np.argsort(t, kind='stable')
            for p, s in zip(pos[order], scores):
                score_map[int(p)] = float(s)
        selected += chosen.tolist()
    return (np.asarray(sorted(selected), int), score_map)


def _lab_tv_distance(before, after):
    labs = np.union1d(before, after)
    if len(before) == 0 or len(after) == 0:
        return np.nan
    p = np.asarray([(before == x).mean() for x in labs])
    q = np.asarray([(after == x).mean() for x in labs])
    return float(0.5 * np.abs(p - q).sum())


def _time_span_ratio(before, after):
    if len(before) < 2 or len(after) < 2:
        return 1.0
    b = float(np.max(before) - np.min(before))
    return 1.0 if b <= 0 else float((np.max(after) - np.min(after)) / b)


def _importance_summary(score_map, selected):
    if not score_map:
        return (np.nan, np.nan, np.nan)
    sel = set(map(int, selected.tolist()))
    all_s = np.asarray(list(score_map.values()), float)
    kept = np.asarray([s for p, s in score_map.items() if p in sel], float)
    removed = np.asarray([s for p, s in score_map.items() if p not in sel], float)
    return (
        float(all_s.mean()),
        float(kept.mean()) if len(kept) else np.nan,
        float(removed.mean()) if len(removed) else np.nan,
    )


def subsample_event_batch(batch, retention, sample_indices, seed, sparse_threshold):
    if not 0 < float(retention) <= 1:
        raise ValueError(f'Invalid retention {retention}')
    missing = {'event_ids', 'values', 'event_times', 'lengths'} - set(batch)
    if missing:
        raise KeyError(f'Missing batch keys: {sorted(missing)}')
    out = _clone_batch(batch)
    ids, vals, times, lengths = (out['event_ids'], out['values'], out['event_times'], out['lengths'])
    mask = out.get('event_mask')
    if ids.ndim != 2 or vals.shape != ids.shape or times.shape != ids.shape:
        raise ValueError('Expected event_ids, values, event_times with identical [B,L] shapes')
    B, Lmax = ids.shape
    if len(sample_indices) != B:
        raise ValueError('sample_indices length does not match batch size')
    diag = []
    for b in range(B):
        L = max(0, min(int(lengths[b].item()), Lmax))
        active = ids[b, :L]
        real_t = torch.nonzero(active >= LAB_ID_OFFSET, as_tuple=False).reshape(-1)
        cls_t = torch.nonzero(active == CLS_ID, as_tuple=False).reshape(-1)
        reg_t = torch.nonzero(active == REG_ID, as_tuple=False).reshape(-1)
        real = real_t.detach().cpu().numpy().astype(int)
        n0 = len(real)
        if n0 == 0:
            diag.append({
                'n_events_original': 0,
                'n_events_kept': 0,
                'realized_retention': np.nan,
                'lab_tv_distance': np.nan,
                'time_span_ratio': 1.0,
                'mean_importance_all': np.nan,
                'mean_importance_kept': np.nan,
                'mean_importance_removed': np.nan,
            })
            continue
        ids0 = ids[b, real_t].detach().cpu().numpy().astype(int)
        vals0 = vals[b, real_t].detach().cpu().numpy().astype(float)
        times0 = times[b, real_t].detach().cpu().numpy().astype(float)
        selected, score_map = _select_patient_events(
            ids0, vals0, times0, real, float(retention),
            int(seed), int(sample_indices[b]), int(sparse_threshold),
        )
        imp_all, imp_kept, imp_removed = _importance_summary(score_map, selected)
        if retention >= 1:
            diag.append({
                'n_events_original': n0,
                'n_events_kept': n0,
                'realized_retention': 1.0,
                'lab_tv_distance': 0.0,
                'time_span_ratio': 1.0,
                'mean_importance_all': imp_all,
                'mean_importance_kept': imp_kept,
                'mean_importance_removed': imp_removed,
            })
            continue
        if len(cls_t) != 1 or len(reg_t) != 1:
            raise RuntimeError(f'sample {sample_indices[b]}: expected one CLS and one REG')
        st = torch.as_tensor(selected, dtype=torch.long, device=ids.device)
        new_len = len(selected) + 2
        if new_len > Lmax:
            raise RuntimeError(f'sample {sample_indices[b]}: compacted length {new_len}>{Lmax}')
        ni, nv, nt = (torch.zeros_like(ids[b]), torch.zeros_like(vals[b]), torch.zeros_like(times[b]))
        cp, rp = (int(cls_t[0]), int(reg_t[0]))
        ni[0], nv[0], nt[0] = (ids[b, cp], vals[b, cp], times[b, cp])
        if len(selected):
            dst = torch.arange(1, 1 + len(selected), device=ids.device)
            ni[dst], nv[dst], nt[dst] = (ids[b, st], vals[b, st], times[b, st])
        rd = 1 + len(selected)
        ni[rd], nv[rd], nt[rd] = (ids[b, rp], vals[b, rp], times[b, rp])
        ids[b], vals[b], times[b], lengths[b] = (ni, nv, nt, new_len)
        if mask is not None:
            nm = torch.zeros_like(mask[b])
            nm[:new_len] = 1
            mask[b] = nm
        kept_ids = ids[b, 1:1 + len(selected)].detach().cpu().numpy().astype(int)
        kept_times = times[b, 1:1 + len(selected)].detach().cpu().numpy().astype(float)
        diag.append({
            'n_events_original': n0,
            'n_events_kept': len(selected),
            'realized_retention': len(selected) / n0,
            'lab_tv_distance': _lab_tv_distance(ids0, kept_ids),
            'time_span_ratio': _time_span_ratio(times0, kept_times),
            'mean_importance_all': imp_all,
            'mean_importance_kept': imp_kept,
            'mean_importance_removed': imp_removed,
        })
    return (out, diag)

@torch.no_grad()
def predict_indices(model, batcher, indices, device, batch_size, retention, seed, sparse_threshold):
    model.eval()
    probs, logits_all, rows = ([], [], [])
    indices = np.asarray(indices, int)
    for start in range(0, len(indices), batch_size):
        ind = indices[start:start + batch_size]
        batch = batcher.get_batch(ind)
        batch, diag = subsample_event_batch(batch, retention, ind, seed, sparse_threshold)
        for i, rec in zip(ind, diag):
            rows.append({'sample_index': int(i), **rec})
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(**batch)
        logits = logits.detach()
        probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        logits_all.append(logits.cpu().numpy().reshape(-1))
    return (np.concatenate(probs), np.concatenate(logits_all), pd.DataFrame(rows))


def seeds_for_retention(r, seeds):
    return [None] if float(r) >= 1 else [int(s) for s in seeds]


def infer_one_fold(cli, fold):
    cp = config_for_fold(fold)
    pt = pretrain_dir_for_fold(cli, fold)
    ft = finetune_dir_for_fold(cli, fold)
    ckpt = ft / 'checkpoint_best.bin'
    require_file(cp, f'fold {fold} config')
    require_file(pt / 'pt_saved_variables.pkl', f'fold {fold} PT metadata')
    require_file(pt / 'checkpoint_best.bin', f'fold {fold} PT checkpoint')
    require_file(ckpt, f'fold {fold} FT checkpoint')
    args = make_pipeline_args(cli, fold, cp, pt)
    env = EnvManager(args)
    ExtractorTrain(env.args)
    env.set_model_params(mode='default')
    env.set_ids(mode='default')
    set_all_seeds(env.args.seed, env.args.fast)
    env.args.cv_mode = 'grid'
    dataset = TimeSeriesDataset(env.args)
    model = build_model(env.args)
    batcher = build_batcher(env.args, dataset.preproc.input_dict)
    model.load_state_dict(torch.load(ckpt, map_location=env.args.device), strict=True)
    model.to(env.args.device).eval()
    test_ind = np.asarray(batcher.splits['test'], int)
    if not len(test_ind):
        raise RuntimeError(f'Fold {fold}: empty test split')
    sup_ids = np.asarray(env.args.ids['sup_ids'])
    hadm = sup_ids[test_ind].astype(np.int64)
    y = np.asarray(batcher.y).reshape(-1)[test_ind].astype(int)
    rows = []
    for retention in cli.retention_levels:
        for s in seeds_for_retention(retention, cli.subsample_seeds):
            seed = 42 if s is None else int(s)
            print(f'  retention={retention:.2f}' + ('' if s is None else f' seed={s}'))
            p, logit, d = predict_indices(
                model, batcher, test_ind, env.args.device,
                cli.infer_batch_size, retention, seed,
                cli.sparse_lab_threshold,
            )
            d = d.set_index('sample_index').loc[test_ind].reset_index()
            rows.append(pd.DataFrame({
                'hadm_id': hadm,
                'fold': fold,
                'label': y,
                'target_retention': retention,
                'seed': np.nan if s is None else s,
                'probability': p,
                'logit': logit,
                'n_events_original': d.n_events_original.to_numpy(),
                'n_events_kept': d.n_events_kept.to_numpy(),
                'realized_retention': d.realized_retention.to_numpy(),
                'lab_tv_distance': d.lab_tv_distance.to_numpy(),
                'time_span_ratio': d.time_span_ratio.to_numpy(),
                'mean_importance_all': d.mean_importance_all.to_numpy(),
                'mean_importance_kept': d.mean_importance_kept.to_numpy(),
                'mean_importance_removed': d.mean_importance_removed.to_numpy(),
            }))
    del model, batcher, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True)


def min_recall_precision(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return np.nan if len(precision) == 0 else float(np.max(np.minimum(precision, recall)))


def metrics_from_arrays(y, p, threshold):
    y, p = (np.asarray(y, int), np.asarray(p, float))
    pred = (p >= threshold).astype(int)
    n, n_pos = (len(y), int(y.sum()))
    out = {
        'n': n,
        'n_pos': n_pos,
        'prevalence': float(y.mean()) if n else np.nan,
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
        out['auroc'] = np.nan
        out['auprc'] = np.nan
        out['minrp'] = np.nan
    return out


def metric_table(pred, threshold, by_fold):
    groups = ['fold', 'target_retention', 'seed'] if by_fold else ['target_retention', 'seed']
    rows = []
    for key, g in pred.groupby(groups, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(groups, key))
        rec.update(metrics_from_arrays(g.label.values, g.probability.values, threshold))
        summary_cols = [
            ('realized_retention', 'mean_realized_retention'),
            ('n_events_original', 'mean_events_original'),
            ('n_events_kept', 'mean_events_kept'),
            ('lab_tv_distance', 'mean_lab_tv_distance'),
            ('time_span_ratio', 'mean_time_span_ratio'),
            ('mean_importance_all', 'mean_importance_all'),
            ('mean_importance_kept', 'mean_importance_kept'),
            ('mean_importance_removed', 'mean_importance_removed'),
        ]
        for src, dst in summary_cols:
            rec[dst] = float(pd.to_numeric(g[src], errors='coerce').mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def add_deltas_vs_original(df):
    out = df.copy()
    cols = ['auroc', 'auprc', 'brier', 'minrp']
    if 'fold' in out:
        orig = (
            out[np.isclose(out.target_retention, 1.0)]
            .set_index('fold')[cols]
            .rename(columns={c: f'original_{c}' for c in cols})
        )
        out = out.join(orig, on='fold')
    else:
        o = out[np.isclose(out.target_retention, 1.0)]
        for c in cols:
            out[f'original_{c}'] = o.iloc[0][c] if len(o) == 1 else np.nan
    for c in cols:
        out[f'delta_{c}_vs_original'] = out[c] - out[f'original_{c}']
    return out


def add_prediction_shift(pred):
    orig = (
        pred[np.isclose(pred.target_retention, 1.0)]
        [['hadm_id', 'fold', 'probability']]
        .drop_duplicates(['hadm_id', 'fold'])
        .rename(columns={'probability': 'original_probability'})
    )
    out = pred.merge(orig, on=['hadm_id', 'fold'], how='left', validate='many_to_one')
    out['delta_probability_vs_original'] = out.probability - out.original_probability
    out['abs_delta_probability_vs_original'] = out.delta_probability_vs_original.abs()
    return out


def probability_shift_summary(pred):
    rows = []
    for (r, s), g in pred.groupby(['target_retention', 'seed'], dropna=False, sort=False):
        x = g.abs_delta_probability_vs_original.dropna().to_numpy(float)
        d = g.delta_probability_vs_original.dropna().to_numpy(float)
        rows.append({
            'target_retention': r,
            'seed': s,
            'n': len(x),
            'mean_abs_prob_shift': np.mean(x) if len(x) else np.nan,
            'median_abs_prob_shift': np.median(x) if len(x) else np.nan,
            'p90_abs_prob_shift': np.percentile(x, 90) if len(x) else np.nan,
            'mean_signed_prob_shift': np.mean(d) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def seed_summary(metrics):
    cols = [
        'auroc', 'auprc', 'brier', 'minrp',
        'delta_auroc_vs_original', 'delta_auprc_vs_original',
        'delta_brier_vs_original', 'delta_minrp_vs_original',
        'mean_realized_retention', 'mean_lab_tv_distance',
        'mean_time_span_ratio', 'mean_importance_all',
        'mean_importance_kept', 'mean_importance_removed',
    ]
    rows = []
    for r, g in metrics.groupby('target_retention', sort=False):
        rec = {'target_retention': float(r), 'n_runs': len(g)}
        for c in cols:
            v = pd.to_numeric(g[c], errors='coerce')
            rec[f'{c}_mean'] = float(v.mean()) if v.notna().any() else np.nan
            rec[f'{c}_std'] = float(v.std(ddof=1)) if v.notna().sum() > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def fold_averaged_summary(by_fold):
    cols = [
        'auroc', 'auprc', 'brier', 'minrp',
        'delta_auroc_vs_original', 'delta_auprc_vs_original',
        'delta_brier_vs_original', 'delta_minrp_vs_original',
        'mean_realized_retention', 'mean_events_kept',
        'mean_lab_tv_distance', 'mean_time_span_ratio',
    ]
    per_fold = by_fold.groupby(['fold', 'target_retention'], as_index=False)[cols].mean()
    rows = []
    for r, g in per_fold.groupby('target_retention', sort=False):
        rec = {'target_retention': float(r), 'n_folds': int(g.fold.nunique())}
        for c in cols:
            v = pd.to_numeric(g[c], errors='coerce')
            rec[f'{c}_mean'] = float(v.mean()) if v.notna().any() else np.nan
            rec[f'{c}_std'] = float(v.std(ddof=1)) if v.notna().sum() > 1 else 0.0
        rows.append(rec)
    return (per_fold, pd.DataFrame(rows))


def main():
    cli = make_cli()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'TASK={TASK} | METHOD={METHOD} | cohort={cli.cohort}')
    print(
        f'Retention={cli.retention_levels} | seeds={cli.subsample_seeds} | '
        f'sparse<= {cli.sparse_lab_threshold} kept'
    )
    if METHOD == 'stratified':
        print(
            f'Stratified: lab x {TIME_BINS} time bins x {VALUE_BINS} value bins | '
            f'endpoints={PROTECT_TIME_ENDPOINTS}'
        )
    else:
        print(f'Adaptive: local-change importance | endpoints={PROTECT_TIME_ENDPOINTS}')
    pred = pd.concat([infer_one_fold(cli, f) for f in cli.folds], ignore_index=True)
    dup = ['hadm_id', 'fold', 'target_retention', 'seed']
    if pred.duplicated(dup).any():
        raise RuntimeError('Duplicate admission/retention predictions detected')
    pred = add_prediction_shift(pred)
    overall = add_deltas_vs_original(metric_table(pred, cli.threshold, False))
    by_fold = add_deltas_vs_original(metric_table(pred, cli.threshold, True))
    shifts = probability_shift_summary(pred)
    seeds = seed_summary(overall)
    per_fold, fold_summary = fold_averaged_summary(by_fold)
    tag = 'event_subsampling' if METHOD == 'stratified' else 'adaptive_subsampling'
    paths = {
        'pred': cli.out_dir / f'oof_{tag}_predictions.csv',
        'overall': cli.out_dir / f'{tag}_metrics_overall.csv',
        'fold': cli.out_dir / f'{tag}_metrics_by_fold.csv',
        'seed': cli.out_dir / f'{tag}_seed_summary.csv',
        'shift': cli.out_dir / f'{tag}_probability_shift_summary.csv',
        'per_fold': cli.out_dir / f'{tag}_seed_averaged_per_fold.csv',
        'fold_summary': cli.out_dir / f'{tag}_fold_averaged_summary.csv',
    }
    outputs = [
        ('pred', pred),
        ('overall', overall),
        ('fold', by_fold),
        ('seed', seeds),
        ('shift', shifts),
        ('per_fold', per_fold),
        ('fold_summary', fold_summary),
    ]
    for k, df in outputs:
        df.to_csv(paths[k], index=False)
    show = [
        'target_retention', 'auroc_mean', 'auroc_std',
        'delta_auroc_vs_original_mean',
        'auprc_mean', 'auprc_std',
        'delta_auprc_vs_original_mean',
    ]
    print('\n===== FOLD-AVERAGED PRIMARY RESULTS =====')
    print(fold_summary[show].to_string(index=False))
    print('\n===== DONE =====')
    for k, p in paths.items():
        print(f'{k:12s}: {p}')

if __name__ == '__main__':
    main()

