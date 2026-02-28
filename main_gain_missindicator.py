# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import data_loader
from gain import gain
from utils import rmse_loss
from datetime import datetime


# -------------------------
# (NEW) helper: build missing-indicator dataset
# -------------------------
def build_missing_indicator_features(miss_data_x, data_m, fill_value=0.0, indicator_is_missing=1):
  """
  miss_data_x: (n,d) with NaN
  data_m:      (n,d) mask, 1=observed, 0=missing  (data_loader 규약)
  fill_value:  NaN을 치환할 값 (downstream 안정성 위해 필요)
  indicator_is_missing:
    - 1이면 indicator=1이 결측(missing), 0이면 observed
    - 0이면 indicator=1이 observed, 0이면 missing (원하면 바꿀 수 있음)
  """
  X_filled = np.array(miss_data_x, copy=True)
  X_filled = np.nan_to_num(X_filled, nan=fill_value)

  if indicator_is_missing == 1:
    ind = 1.0 - data_m   # 1=missing
  else:
    ind = data_m         # 1=observed

  X_with_ind = np.concatenate([X_filled, ind], axis=1)
  return X_filled, ind, X_with_ind


def main(args):
  data_name = args.data_name
  miss_rate = args.miss_rate

  gain_parameters = {
    'batch_size': args.batch_size,
    'hint_rate': args.hint_rate,
    'alpha': args.alpha,
    'iterations': args.iterations,
    'log_every': 100,
    'verbose': False
  }

  # -------------------------
  # (NEW) run_name에 모드 반영
  # -------------------------
  run_name = f"{data_name}_mcar{miss_rate:.2f}_bs{args.batch_size}_alpha{args.alpha}_{args.mode}"

  wandb_run = wandb.init(
    project="HK_GAIN",
    name=run_name,
    config={
      "data_name": data_name,
      "miss_rate": miss_rate,
      "batch_size": args.batch_size,
      "hint_rate": args.hint_rate,
      "alpha": args.alpha,
      "iterations": args.iterations,
      "log_every": gain_parameters['log_every'],
      "verbose": gain_parameters['verbose'],
      "mode": args.mode,
      "fill_value": args.fill_value,
      "indicator_is_missing": args.indicator_is_missing,
    }
  )

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_dir = os.path.join("_outputs", f"{run_name}_{timestamp}")
  os.makedirs(run_dir, exist_ok=True)

  imp_dir = os.path.join(run_dir, "imputed_data")
  os.makedirs(imp_dir, exist_ok=True)

  print("Experiment directory:", run_dir)

  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, y, ids = data_loader(data_name, miss_rate)

  # -------------------------
  # (NEW) mode 분기
  # -------------------------
  imputed_data_x = None
  rmse = None

  if args.mode in ["gain", "both"]:
    # (기존) GAIN으로 imputation
    imputed_data_x = gain(miss_data_x, gain_parameters, wandb_run=wandb_run)
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    print('\nRMSE Performance: ' + str(np.round(rmse, 4)))
    wandb_run.log({"final/RMSE": float(rmse)})

    np.save(os.path.join(imp_dir, "imputed.npy"), imputed_data_x)

  if args.mode in ["indicator", "both"]:
    # (NEW) missing-indicator용 feature 생성
    X_filled, miss_ind, X_with_ind = build_missing_indicator_features(
      miss_data_x=miss_data_x,
      data_m=data_m,
      fill_value=args.fill_value,
      indicator_is_missing=args.indicator_is_missing
    )

    # 저장: downstream이 바로 쓰기 좋게 명확히 분리
    np.save(os.path.join(imp_dir, "missing_filled.npy"), X_filled)                 # (n,d)
    np.save(os.path.join(imp_dir, "missing_indicator.npy"), miss_ind)             # (n,d)
    np.save(os.path.join(imp_dir, "missing_indicator_concat.npy"), X_with_ind)    # (n,2d)

    # wandb에 shape만 로깅 (성능은 downstream에서)
    wandb_run.log({
      "final/indicator_shape_n": int(X_with_ind.shape[0]),
      "final/indicator_shape_d": int(X_with_ind.shape[1]),
    })

  # 공통 저장 (기존과 동일)
  np.save(os.path.join(imp_dir, "original.npy"), ori_data_x)
  np.save(os.path.join(imp_dir, "mask.npy"), data_m)
  np.save(os.path.join(imp_dir, "labels.npy"), y)
  np.save(os.path.join(imp_dir, "ids.npy"), ids)

  wandb.finish()

  return imputed_data_x, rmse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_name', choices=['letter','spam', 'breastcancer'], default='breastcancer', type=str)
  parser.add_argument('--miss_rate', default=0.2, type=float)
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--hint_rate', default=0.9, type=float)
  parser.add_argument('--alpha', default=100, type=float)
  parser.add_argument('--iterations', default=10000, type=int)

  # -------------------------
  # (NEW) 모드 추가
  # -------------------------
  parser.add_argument(
    '--mode',
    choices=['gain', 'indicator', 'both'],
    default='both',
    help="gain: 기존 GAIN imputation만 / indicator: missing indicator만 / both: 둘 다 저장"
  )
  parser.add_argument(
    '--fill_value',
    type=float,
    default=0.0,
    help="indicator 모드에서 NaN을 치환할 값 (downstream 안정성 목적)"
  )
  parser.add_argument(
    '--indicator_is_missing',
    type=int,
    choices=[0,1],
    default=1,
    help="1이면 indicator=1이 missing, 0이면 indicator=1이 observed"
  )

  args = parser.parse_args()
  imputed_data, rmse = main(args)