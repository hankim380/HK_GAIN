# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')  # For headless environments (no display)
import matplotlib.pyplot as plt

from data_loader import data_loader
from gain import gain
from utils import rmse_loss

from datetime import datetime


def log_imputation_diagnostics(
    ori_data_x,
    miss_data_x,
    imputed_data_x,
    data_m,
    wandb_run,
    run_dir,
    max_hist_features=20,
    hist_bins=50,
    corr_max_dim=200,
):
  """
  ori_data_x: (n,d) 완전 데이터 (정답)
  miss_data_x: (n,d) 결측 포함 데이터 (NaN)
  imputed_data_x: (n,d) GAIN 출력 (결측 채움)
  data_m: (n,d) indicator (1=observed, 0=missing)  <-- data_loader.py 규약
  wandb_run: wandb run
  """

  if wandb_run is None:
    return
  
  # 저장 폴더 생성
  hist_dir = os.path.join(run_dir, "hist_missing_only")
  corr_dir = os.path.join(run_dir, "corr")
  os.makedirs(hist_dir, exist_ok=True)
  os.makedirs(corr_dir, exist_ok=True)

  n, d = ori_data_x.shape
  missing_mask = (data_m == 0)

  # (A) 변수별 histogram 비교

  # 전체 변수를 다 올리면 과함. (spam은 57개라 가능하지만 기본은 제한)
  num_features = min(d, max_hist_features)

  for j in range(num_features):
    # 결측이 실제로 발생한 위치에서만 비교하는 게 핵심 (missing-only)
    miss_j = missing_mask[:, j]
    if miss_j.sum() == 0:
      # 해당 변수는 결측이 없으면 히스토그램 비교 의미가 약함 -> 스킵(원하면 전체 비교로 바꿀 수 있음)
      continue

    gt = ori_data_x[miss_j, j]
    imp = imputed_data_x[miss_j, j]

    fig = plt.figure()
    plt.hist(gt, bins=hist_bins, alpha=0.5, label="GT (missing positions)")
    plt.hist(imp, bins=hist_bins, alpha=0.5, label="Imputed (missing positions)")
    plt.title(f"Feature {j}: distribution @ missing positions")
    plt.legend()

    # 로컬 저장
    fig_path = os.path.join(hist_dir, f"feature_{j}.png")
    fig.savefig(fig_path, bbox_inches="tight")

    # wandb 업로드
    wandb_run.log({f"diagnostics/hist_missing_only/feature_{j}": wandb.Image(fig)})
    plt.close(fig)


  # (B) Correlation matrix + difference heatmap

  # 고차원(d가 너무 크면) correlation heatmap은 의미도 약하고 이미지도 너무 큼
  # spam/letter는 보통 충분히 작지만, 안전장치로 차원 제한
  d_corr = min(d, corr_max_dim)
  X_gt = ori_data_x[:, :d_corr]
  X_imp = imputed_data_x[:, :d_corr]

  # 상관계수 계산 (열 기준)
  # 주의: 상수열이 있으면 NaN이 뜰 수 있음 -> nan_to_num으로 처리
  corr_gt = np.corrcoef(X_gt, rowvar=False)
  corr_imp = np.corrcoef(X_imp, rowvar=False)

  corr_gt = np.nan_to_num(corr_gt, nan=0.0, posinf=0.0, neginf=0.0)
  corr_imp = np.nan_to_num(corr_imp, nan=0.0, posinf=0.0, neginf=0.0)

  corr_diff = corr_imp - corr_gt

  # GT corr heatmap
  fig1 = plt.figure()
  plt.imshow(corr_gt, aspect="auto")
  plt.title(f"Correlation (GT)  d={d_corr}")
  plt.colorbar()
  wandb_run.log({"diagnostics/corr/gt": wandb.Image(fig1)})
  plt.close(fig1)

  # Imputed corr heatmap
  fig2 = plt.figure()
  plt.imshow(corr_imp, aspect="auto")
  plt.title(f"Correlation (Imputed)  d={d_corr}")
  plt.colorbar()
  wandb_run.log({"diagnostics/corr/imputed": wandb.Image(fig2)})
  plt.close(fig2)

  # Diff heatmap
  fig3 = plt.figure()
  plt.imshow(corr_diff, aspect="auto")
  plt.title(f"Correlation Difference (Imputed - GT)  d={d_corr}")
  plt.colorbar()
  wandb_run.log({"diagnostics/corr/diff": wandb.Image(fig3)})
  plt.close(fig3)

  # 추가로 숫자 요약도 남기면 좋음(대시보드에서 비교 쉬움)
  wandb_run.log({
    "diagnostics/corr_diff_abs_mean": float(np.mean(np.abs(corr_diff))),
    "diagnostics/corr_diff_abs_max": float(np.max(np.abs(corr_diff))),
  })



def log_featurewise_rmse(
    ori_data_x,
    imputed_data_x,
    data_m,
    wandb_run,
    run_dir
):
  """
  변수별 RMSE (missing 위치에 대해서만 계산)
  """

  if wandb_run is None:
    return
  
  # 저장 폴더
  imp_dir = os.path.join(run_dir, "imputed_data")
  os.makedirs(imp_dir, exist_ok=True)

  n, d = ori_data_x.shape
  missing_mask = (data_m == 0)

  feature_rmses = []

  for j in range(d):
    miss_j = missing_mask[:, j]

    if miss_j.sum() == 0:
      feature_rmses.append(np.nan)
      continue

    gt = ori_data_x[miss_j, j]
    imp = imputed_data_x[miss_j, j]

    rmse_j = np.sqrt(np.mean((gt - imp) ** 2))
    feature_rmses.append(float(rmse_j))

  feature_rmses = np.array(feature_rmses)

  # wandb scalar logging
  wandb_run.log({
      "diagnostics/feature_rmse_mean": float(np.mean(feature_rmses)),
      "diagnostics/feature_rmse_max": float(np.max(feature_rmses))
  })

  # 로컬에 값 저장
  np.save(os.path.join(imp_dir, "feature_rmse.npy"), feature_rmses)

  # Bar plot 생성
  fig = plt.figure(figsize=(10, 4))
  plot_vals = np.nan_to_num(feature_rmses, nan=0.0)
  plt.bar(np.arange(d), plot_vals)
  plt.title("Feature-wise RMSE (missing positions only)")
  plt.xlabel("Feature Index")
  plt.ylabel("RMSE")

  wandb_run.log({
  "diagnostics/feature_rmse_mean": float(np.nanmean(feature_rmses)),
  "diagnostics/feature_rmse_max": float(np.nanmax(feature_rmses))
  })
  plt.close(fig)

  return feature_rmses




def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''

  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'log_every': 100,
                     'verbose': False}
  
  # WandB INIT (실험 단위 관리)
  run_name = f"{data_name}_mcar{miss_rate:.2f}_bs{args.batch_size}_alpha{args.alpha}"

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
          "verbose": gain_parameters['verbose']
      }
  )

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_dir = os.path.join("_outputs", f"{run_name}_{timestamp}")
  os.makedirs(run_dir, exist_ok=True)

  # 하위 폴더 생성
  imp_dir = os.path.join(run_dir, "imputed_data")
  corr_dir = os.path.join(run_dir, "corr")
  hist_dir = os.path.join(run_dir, "hist_missing_only")
  os.makedirs(imp_dir, exist_ok=True)
  os.makedirs(corr_dir, exist_ok=True)
  os.makedirs(hist_dir, exist_ok=True)

  print("Experiment directory:", run_dir)
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, y, ids = data_loader(data_name, miss_rate)
  
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters,
                        wandb_run=wandb_run)  # Pass the wandb run to gain
  
  # Report the RMSE performance
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))

  np.save(os.path.join(imp_dir, "imputed.npy"), imputed_data_x)
  np.save(os.path.join(imp_dir, "original.npy"), ori_data_x)
  np.save(os.path.join(imp_dir, "mask.npy"), data_m)
  np.save(os.path.join(imp_dir, "labels.npy"), y)
  np.save(os.path.join(imp_dir, "ids.npy"), ids)

  # wandb에 최종 metric 기록
  wandb_run.log({"final/RMSE": float(rmse)})

  # Additional diagnostics
  log_imputation_diagnostics(
      ori_data_x=ori_data_x,
      miss_data_x=miss_data_x,
      imputed_data_x=imputed_data_x,
      data_m=data_m,
      wandb_run=wandb_run,
      run_dir=run_dir,
      max_hist_features=20,
      hist_bins=50,
      corr_max_dim=200
  )

  # 변수별 RMSE 추가
  log_featurewise_rmse(
      ori_data_x,
      imputed_data_x,
      data_m,
      wandb_run,
      run_dir=run_dir
  )

  wandb.finish()
  
  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'breastcancer'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
