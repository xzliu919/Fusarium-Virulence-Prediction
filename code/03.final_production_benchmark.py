#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import h5py
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ==========================================
# 1. 深度学习架构库 (PyTorch)
# ==========================================
class Net_MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x.squeeze(1))

class Net_PNNGS(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.3), nn.Flatten(),
            nn.Linear(16 * in_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class Net_ResGS(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1, self.bn1, self.relu = nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16), nn.ReLU()
        self.conv2, self.bn2 = nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.BatchNorm1d(16)
        self.fc1, self.fc2, self.dropout = nn.Linear(16 * in_features, 64), nn.Linear(64, 1), nn.Dropout(0.3)
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        res = x1 + self.relu(self.bn2(self.conv2(x1)))
        return self.fc2(self.dropout(self.relu(self.fc1(res.view(res.size(0), -1)))))

class Net_TabAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.latent_tokens, self.embed_dim = 64, 16
        self.tokenizer = nn.Linear(in_features, self.latent_tokens * self.embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(self.latent_tokens * self.embed_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = x.squeeze(1) if x.dim() == 3 else x
        tokens = self.tokenizer(x).view(x.size(0), self.latent_tokens, self.embed_dim)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        return self.fc(attn_out)

def train_dl_model(X_train, y_train, X_test, epochs, batch_size, arch_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_features = X_train.shape[1]
    if arch_name == 'MLP': model = Net_MLP(in_features)
    elif arch_name == 'PNNGS': model = Net_PNNGS(in_features)
    elif arch_name == 'ResGS': model = Net_ResGS(in_features)
    elif arch_name == 'TabAttention': model = Net_TabAttention(in_features)
    
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model = model.to(device)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer, criterion = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad(): preds = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy().squeeze()
    return np.expand_dims(preds, 0) if preds.ndim == 0 else preds

# --- 新增：用于在全量数据上训练冠军模型并返回模型实例 ---
def train_final_dl_model(X_train, y_train, epochs, batch_size, arch_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_features = X_train.shape[1]
    if arch_name == 'MLP': model = Net_MLP(in_features)
    elif arch_name == 'PNNGS': model = Net_PNNGS(in_features)
    elif arch_name == 'ResGS': model = Net_ResGS(in_features)
    elif arch_name == 'TabAttention': model = Net_TabAttention(in_features)
    
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model = model.to(device)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer, criterion = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
            
    return model

# ==========================================
# 2. 核心工具函数与主流程
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pav', type=str, default="data/pav_matrix.csv")
    parser.add_argument('--cnv', type=str, default="data/cnv_matrix.csv")
    parser.add_argument('--h5_in', type=str, default="data/protein_embeddings.h5")
    parser.add_argument('--og_map', type=str, default="data/og_mapping.tsv")
    parser.add_argument('--metadata', type=str, default="data/metadata.csv")
    parser.add_argument('--phylo_dist', type=str, default=None)
    parser.add_argument('--phylo_dim', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='../results_final_production')
    parser.add_argument('--phenotypes', nargs='+', default=['Phenotype_WheatHead', 'Phenotype_WheatStem', 'Phenotype_MaizeStem', 'Phenotype_SoybeanStem'])
    parser.add_argument('--k_grid', nargs='+', type=int, default=[200, 500, 1000])
    parser.add_argument('--pca_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128) # 大Batch适配多卡
    parser.add_argument('--models', nargs='+', default=['Ridge', 'SVR', 'RandomForest', 'GBR', 'MLP', 'PNNGS', 'ResGS', 'TabAttention'])
    return parser.parse_args()

def build_og_dict(path):
    d = defaultdict(lambda: defaultdict(list))
    for _, r in pd.read_csv(path, sep='\t').iterrows(): d[r['Species']][r['OG_ID']].append(r['Protein_ID'])
    return d

def calc_metrics(y_true, y_pred):
    if len(y_true) < 2: return {'R2': 0.0, 'Pearson_r': 0.0, 'Spearman_rho': 0.0, 'RMSE': np.nan, 'MAE': np.nan}
    return {
        'R2': r2_score(y_true, y_pred), 'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 'MAE': mean_absolute_error(y_true, y_pred),
        'Pearson_r': 0.0 if np.std(y_pred)==0 or np.std(y_true)==0 else pearsonr(y_true, y_pred)[0],
        'Spearman_rho': 0.0 if np.std(y_pred)==0 or np.std(y_true)==0 else spearmanr(y_true, y_pred)[0]
    }

def extract_fused_features(target_ogs, pav_df, cnv_df, species_list, h5_path, og_map_dict, pca_dim):
    X_pav, X_cnv = pav_df[target_ogs].values, cnv_df[target_ogs].values
    valid_embs, valid_idxs = [], []
    with h5py.File(h5_path, 'r') as h5_file:
        for i, sp in enumerate(species_list):
            if sp not in h5_file: continue
            for j, og in enumerate(target_ogs):
                if pav_df.at[sp, og] == 0: continue
                emb_list = [h5_file[sp][p][:] for p in og_map_dict[sp].get(og, []) if p in h5_file[sp]]
                if emb_list: valid_embs.append(np.mean(emb_list, axis=0)); valid_idxs.append((i, j))
    emb_tensor = np.zeros((len(species_list), len(target_ogs), pca_dim), dtype=np.float32)
    for (i, j), red_emb in zip(valid_idxs, PCA(n_components=pca_dim, random_state=42).fit_transform(np.array(valid_embs))): emb_tensor[i, j] = red_emb
    return np.concatenate([X_pav[..., np.newaxis], X_cnv[..., np.newaxis], emb_tensor], axis=-1).reshape((len(species_list), -1))

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "saved_models"), exist_ok=True)
    HOLDOUT_SPECIES = ["Fusarium_aberrans__CBS_119866", "Fusarium_californicum__CBS_145796", "Fusarium_languescens__CBS_645.78", "Fusarium_pseudograminearum__CS3096", "Fusarium_flocciferum__CBS_821.68"]
    
    print(f"[*] 初始化 8 模型终极生产流水线 | 多卡并行: {torch.cuda.device_count()} GPUs...")
    pav_df, cnv_df, meta_df = pd.read_csv(args.pav, index_col=0), pd.read_csv(args.cnv, index_col=0), pd.read_csv(args.metadata, index_col=0)
    if pav_df.shape[0] > pav_df.shape[1]: pav_df, cnv_df = pav_df.T, cnv_df.T
    common_species = list(set(pav_df.index) & set(cnv_df.index) & set(meta_df.index))
    
    phylo_emb_df = None
    if args.phylo_dist and os.path.exists(args.phylo_dist):
        common_species = list(set(common_species) & set(pd.read_csv(args.phylo_dist, index_col=0).index))
        dist_matrix = pd.read_csv(args.phylo_dist, index_col=0).loc[common_species, common_species].values
        phylo_emb_df = pd.DataFrame(MDS(n_components=args.phylo_dim, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix), index=common_species)
    
    pav_df, cnv_df, meta_df = pav_df.loc[common_species], cnv_df.loc[common_species], meta_df.loc[common_species]
    og_names, species_array, og_map_dict = pav_df.columns.tolist(), np.array(common_species), build_og_dict(args.og_map)
    base_ml_dict = {'Ridge': Ridge(alpha=10.0), 'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1), 'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), 'GBR': GradientBoostingRegressor(n_estimators=100, random_state=42)}
    
    all_cv_records, all_ho_records = [], []

    for pheno in args.phenotypes:
        if pheno not in meta_df.columns or not np.any(~np.isnan(meta_df[pheno].values)): continue
        valid_idx = ~np.isnan(meta_df[pheno].values)
        y, curr_species = meta_df[pheno].values[valid_idx], species_array[valid_idx]
        train_mask, test_mask = ~np.isin(curr_species, HOLDOUT_SPECIES), np.isin(curr_species, HOLDOUT_SPECIES)
        y_tr, y_te = y[train_mask], y[test_mask]
        X_pav_tr, X_cnv_tr = pav_df.iloc[valid_idx].iloc[train_mask].values, cnv_df.iloc[valid_idx].iloc[train_mask].values
        phylo_tr = phylo_emb_df.iloc[valid_idx].iloc[train_mask].values if phylo_emb_df is not None else None
        phylo_te = phylo_emb_df.iloc[valid_idx].iloc[test_mask].values if phylo_emb_df is not None else None

        print(f"\n{'='*70}\n[>>>] {pheno} | Train: {len(y_tr)} | Test: {len(y_te)}\n{'='*70}")
        y_tr_res = y_tr - Ridge(alpha=10.0).fit(phylo_tr, y_tr).predict(phylo_tr) if phylo_tr is not None else y_tr
        rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1).fit(np.hstack([X_pav_tr, X_cnv_tr]), y_tr_res)
        sorted_ogs = np.argsort(rf.feature_importances_[:len(og_names)] + rf.feature_importances_[len(og_names):])[::-1]
        
        best_overall_r, best_overall_model, best_overall_k = -float('inf'), None, None
        
        for model_name in args.models:
            print(f"\n--- {model_name} ---")
            best_k, best_k_r = args.k_grid[0], -float('inf')
            
            # Phase 1: K Grid Search
            for k in args.k_grid:
                X_tr_k = np.hstack([X_pav_tr[:, sorted_ogs[:k]], X_cnv_tr[:, sorted_ogs[:k]]])
                y_cv_preds = np.zeros_like(y_tr)
                for tr_idx, va_idx in KFold(n_splits=10, shuffle=True, random_state=42).split(X_tr_k):
                    f_res_tr = y_tr[tr_idx] - Ridge(alpha=10.0).fit(phylo_tr[tr_idx], y_tr[tr_idx]).predict(phylo_tr[tr_idx]) if phylo_tr is not None else y_tr[tr_idx]
                    f_phy_va = Ridge(alpha=10.0).fit(phylo_tr[tr_idx], y_tr[tr_idx]).predict(phylo_tr[va_idx]) if phylo_tr is not None else np.zeros_like(y_tr[va_idx])
                    Xc_tr, Xc_va = StandardScaler().fit_transform(X_tr_k[tr_idx]), StandardScaler().fit(X_tr_k[tr_idx]).transform(X_tr_k[va_idx])
                    
                    if model_name in base_ml_dict: f_pred = clone(base_ml_dict[model_name]).fit(Xc_tr, f_res_tr).predict(Xc_va)
                    else: f_pred = train_dl_model(Xc_tr, f_res_tr, Xc_va, 50, args.batch_size, model_name)
                    y_cv_preds[va_idx] = np.maximum(f_phy_va + f_pred, 0)
                
                r_val = calc_metrics(y_tr, y_cv_preds)['Pearson_r']
                if r_val > best_k_r: best_k_r, best_k = r_val, k

            # Phase 2: Final Eval with ESM3
            print(f"  [+] Optimal K={best_k}. Fusing ESM3...")
            X_fused = extract_fused_features([og_names[i] for i in sorted_ogs[:best_k]], pav_df.iloc[valid_idx], cnv_df.iloc[valid_idx], curr_species, args.h5_in, og_map_dict, args.pca_dim)
            X_fused_tr, X_fused_te = X_fused[train_mask], X_fused[test_mask]
            
            # 10-Fold
            for fold_i, (tr_idx, va_idx) in enumerate(KFold(n_splits=10, shuffle=True, random_state=42).split(X_fused_tr)):
                f_res_tr = y_tr[tr_idx] - Ridge(alpha=10.0).fit(phylo_tr[tr_idx], y_tr[tr_idx]).predict(phylo_tr[tr_idx]) if phylo_tr is not None else y_tr[tr_idx]
                f_phy_va = Ridge(alpha=10.0).fit(phylo_tr[tr_idx], y_tr[tr_idx]).predict(phylo_tr[va_idx]) if phylo_tr is not None else np.zeros_like(y_tr[va_idx])
                Xc_tr, Xc_va = StandardScaler().fit_transform(X_fused_tr[tr_idx]), StandardScaler().fit(X_fused_tr[tr_idx]).transform(X_fused_tr[va_idx])
                
                if model_name in base_ml_dict: f_pred = clone(base_ml_dict[model_name]).fit(Xc_tr, f_res_tr).predict(Xc_va)
                else: f_pred = train_dl_model(Xc_tr, f_res_tr, Xc_va, args.epochs, args.batch_size, model_name)
                
                f_mets = calc_metrics(y_tr[va_idx], np.maximum(f_phy_va + f_pred, 0))
                f_mets.update({'Phenotype': pheno, 'Model': model_name, 'Fold': fold_i + 1})
                all_cv_records.append(f_mets)
                
            # Holdout
            res_tr_fin = y_tr - Ridge(alpha=10.0).fit(phylo_tr, y_tr).predict(phylo_tr) if phylo_tr is not None else y_tr
            phy_pred_te = Ridge(alpha=10.0).fit(phylo_tr, y_tr).predict(phylo_te) if phylo_tr is not None else np.zeros_like(y_te)
            Xf_tr_s, Xf_te_s = StandardScaler().fit_transform(X_fused_tr), StandardScaler().fit(X_fused_tr).transform(X_fused_te)
            
            if model_name in base_ml_dict: res_pred_te = clone(base_ml_dict[model_name]).fit(Xf_tr_s, res_tr_fin).predict(Xf_te_s)
            else: res_pred_te = train_dl_model(Xf_tr_s, res_tr_fin, Xf_te_s, args.epochs, args.batch_size, model_name)
            
            ho_mets = calc_metrics(y_te, np.maximum(phy_pred_te + res_pred_te, 0))
            ho_mets.update({'Phenotype': pheno, 'Model': model_name})
            all_ho_records.append(ho_mets)
            print(f"      -> Holdout Pearson r: {ho_mets['Pearson_r']:.4f}")
            
            if ho_mets['Pearson_r'] > best_overall_r:
                best_overall_r, best_overall_model, best_overall_k = ho_mets['Pearson_r'], model_name, best_k

        # ==========================================================
        # 冠军模型全量数据训练与保存 (包含深度学习权重)
        # ==========================================================
        print(f"\n[🏆] {pheno} 冠军模型: {best_overall_model} (K={best_overall_k})")
        top_ogs = [og_names[i] for i in sorted_ogs[:best_overall_k]]
        X_all_fused = extract_fused_features(top_ogs, pav_df.iloc[valid_idx], cnv_df.iloc[valid_idx], curr_species, args.h5_in, og_map_dict, args.pca_dim)
        
        phylo_all_model = Ridge(alpha=10.0).fit(phylo_emb_df.iloc[valid_idx].values, y) if phylo_emb_df is not None else None
        y_res_all = y - phylo_all_model.predict(phylo_emb_df.iloc[valid_idx].values) if phylo_all_model else y
        
        final_scaler = StandardScaler()
        X_all_fused_s = final_scaler.fit_transform(X_all_fused)
        
        dl_weight_path = None
        if best_overall_model not in base_ml_dict:
            print(f"  [+] 正在全量数据上重新训练深度学习冠军模型: {best_overall_model} ...")
            final_dl_model = train_final_dl_model(X_all_fused_s, y_res_all, args.epochs, args.batch_size, best_overall_model)
            saved_fn = "DeepLearning_PyTorch_Model"
            dl_weight_path = os.path.join(args.outdir, "saved_models", f"{pheno}_{best_overall_model}_weights.pt")
            
            # 如果是多卡训练，保存前解包 (去掉 module. 前缀)，方便后续 CPU 推理
            model_to_save = final_dl_model.module if isinstance(final_dl_model, nn.DataParallel) else final_dl_model
            torch.save(model_to_save.state_dict(), dl_weight_path)
            print(f"  [✔] 深度学习权重已独立保存至: {dl_weight_path}")
        else:
            saved_fn = clone(base_ml_dict[best_overall_model]).fit(X_all_fused_s, y_res_all)
        
        joblib.dump({
            'phenotype': pheno, 'top_ogs': top_ogs, 'phylo_baseline_model': phylo_all_model,
            'scaler': final_scaler, 'functional_model_name': best_overall_model, 
            'functional_model': saved_fn,
            'dl_weight_file': dl_weight_path # 记录深度学习权重路径
        }, os.path.join(args.outdir, "saved_models", f"{pheno}_Production_Pipeline.pkl"))

    pd.DataFrame(all_cv_records).to_csv(os.path.join(args.outdir, "cv_fold_metrics.csv"), index=False)
    pd.DataFrame(all_ho_records).to_csv(os.path.join(args.outdir, "holdout_metrics.csv"), index=False)
    print(f"\n[🎉] 全部 8 种模型验证完毕并已覆盖保存至 {args.outdir}")

if __name__ == '__main__':
    main()
