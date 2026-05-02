#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import h5py
import warnings
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")

# ==========================================
# 1. 深度学习推断架构库 (需与训练时保持一致)
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


# ---> 这是关键修改点：支持加载真实的 .pt 权重文件 <---
def inference_dl_model(X_test, arch_name, weight_path):
    # 因为只是简单的前向推断，统一用 CPU 即可
    device = torch.device('cpu')
    in_features = X_test.shape[1]
    
    if arch_name == 'PNNGS': model = Net_PNNGS(in_features)
    elif arch_name == 'ResGS': model = Net_ResGS(in_features)
    elif arch_name == 'MLP': model = Net_MLP(in_features)
    elif arch_name == 'TabAttention': model = Net_TabAttention(in_features)
    else: raise ValueError(f"Unknown DL architecture for inference: {arch_name}")
    
    # 智能加载权重：处理之前如果用了 nn.DataParallel 训练保存的权重
    state_dict = torch.load(weight_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy().squeeze()
    return np.expand_dims(preds, 0) if preds.ndim == 0 else preds

# ==========================================
# 2. 核心工具函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Prediction on Unknown Species")
    parser.add_argument('--pav', type=str, default="data/pav_matrix.csv")
    parser.add_argument('--cnv', type=str, default="data/cnv_matrix.csv")
    parser.add_argument('--h5_in', type=str, default="data/protein_embeddings.h5")
    parser.add_argument('--og_map', type=str, default="data/og_mapping.tsv")
    # 距离矩阵必须包含未知物种！
    parser.add_argument('--phylo_dist', type=str, default=None)
    parser.add_argument('--phylo_dim', type=int, default=10)
    
    # 之前保存的生产级模型所在的目录
    parser.add_argument('--model_dir', type=str, default='./results/saved_models')
    parser.add_argument('--outdir', type=str, default='./results')
    
    # 我们知道的、已经有表型的 88 个物种的列表，用于排除，只留下“未知”物种
    parser.add_argument('--metadata', type=str, default="data/metadata.csv")
    parser.add_argument('--pca_dim', type=int, default=16)
    return parser.parse_args()

def build_og_dict(path):
    d = defaultdict(lambda: defaultdict(list))
    for _, r in pd.read_csv(path, sep='\t').iterrows(): d[r['Species']][r['OG_ID']].append(r['Protein_ID'])
    return d

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
    if valid_embs:
        pca = PCA(n_components=pca_dim, random_state=42)
        reduced_embeddings = pca.fit_transform(np.array(valid_embs))
        for (i, j), red_emb in zip(valid_idxs, reduced_embeddings): 
            emb_tensor[i, j] = red_emb
    else:
        print("  [!] 警告: 当前未知物种批次中未能提取到任何有效的靶点 ESM3 特征，将使用全零矩阵替补。")
        
    return np.concatenate([X_pav[..., np.newaxis], X_cnv[..., np.newaxis], emb_tensor], axis=-1).reshape((len(species_list), -1))

# ==========================================
# 3. 主预测流程
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    print("[*] 开始加载端到端高通量虚拟表型推断流水线...")
    
    # 1. 加载全量基因组数据
    pav_df = pd.read_csv(args.pav, index_col=0)
    cnv_df = pd.read_csv(args.cnv, index_col=0)
    if pav_df.shape[0] > pav_df.shape[1]: pav_df, cnv_df = pav_df.T, cnv_df.T
    
    # 2. 锁定“未知表型”的物种名单
    all_genomic_species = list(set(pav_df.index) & set(cnv_df.index))
    meta_df = pd.read_csv(args.metadata, index_col=0)
    known_species = set(meta_df.index)
    
    unknown_species = sorted(list(set(all_genomic_species) - known_species))
    
    if not unknown_species:
        print("[!] 没有找到需要预测的未知物种，程序退出。")
        return
        
    print(f"[*] 共发现 {len(unknown_species)} 个未鉴定表型的候选镰刀菌物种。")
    
    # 3. 准备系统发育距离矩阵
    phylo_emb_df = None
    if args.phylo_dist and os.path.exists(args.phylo_dist):
        dist_df = pd.read_csv(args.phylo_dist, index_col=0)
        valid_phylo_species = list(set(all_genomic_species) & set(dist_df.index))
        dist_matrix = dist_df.loc[valid_phylo_species, valid_phylo_species].values
        mds = MDS(n_components=args.phylo_dim, dissimilarity='precomputed', random_state=42)
        phylo_emb_df = pd.DataFrame(mds.fit_transform(dist_matrix), index=valid_phylo_species)
        print(f"  [+] 成功为 {len(valid_phylo_species)} 个物种构建了 {args.phylo_dim} 维系统发育演化基线坐标。")

    # 4. 加载基因映射字典
    print("  [+] 正在加载靶点蛋白质映射关系字典...")
    og_map_dict = build_og_dict(args.og_map)
    
    final_predictions_df = pd.DataFrame(index=unknown_species)
    
    # 5. 遍历四个组织，加载对应的冠军模型进行预测
    phenotypes = ['Phenotype_WheatHead', 'Phenotype_WheatStem', 'Phenotype_MaizeStem', 'Phenotype_SoybeanStem']
    
    for pheno in phenotypes:
        model_path = os.path.join(args.model_dir, f"{pheno}_Production_Pipeline.pkl")
        if not os.path.exists(model_path):
            print(f"\n[!] 找不到 {pheno} 的生产级模型包: {model_path}，跳过该表型预测。")
            continue
            
        print(f"\n{'='*70}")
        print(f"[>>>] 正在对未知物种推断表型: {pheno}")
        print(f"{'='*70}")
        
        # 拆解模型包裹
        pipeline = joblib.load(model_path)
        top_ogs = pipeline['top_ogs']
        phylo_baseline_model = pipeline['phylo_baseline_model']
        final_scaler = pipeline['scaler']
        model_name = pipeline['functional_model_name']
        functional_model = pipeline['functional_model']
        
        print(f"  -> 使用冠军架构: [{model_name}], 依赖核心功能基因 (K): {len(top_ogs)} 个")
        
        curr_predict_species = unknown_species
        if phylo_emb_df is not None and phylo_baseline_model is not None:
            curr_predict_species = [s for s in unknown_species if s in phylo_emb_df.index]
            print(f"  -> 其中 {len(curr_predict_species)} 个物种具备系统发育特征，可执行两阶段推断。")
        
        if not curr_predict_species:
            print("  [!] 候选物种缺乏必需的特征输入，跳过。")
            continue
            
        print("  -> 正在提取并拼接目标物种的 PAV, CNV 及 ESM-2 序列结构变异特征...")
        X_fused_unknown = extract_fused_features(
            target_ogs=top_ogs, 
            pav_df=pav_df.loc[curr_predict_species], 
            cnv_df=cnv_df.loc[curr_predict_species], 
            species_list=curr_predict_species, 
            h5_path=args.h5_in, 
            og_map_dict=og_map_dict, 
            pca_dim=args.pca_dim
        )
        
        X_fused_unknown_s = final_scaler.transform(X_fused_unknown)
        
        print("  -> 执行双模态解耦推断 (Base + Residual)...")
        if phylo_emb_df is not None and phylo_baseline_model is not None:
            phylo_feat_unknown = phylo_emb_df.loc[curr_predict_species].values
            base_pred = phylo_baseline_model.predict(phylo_feat_unknown)
        else:
            base_pred = np.zeros(len(curr_predict_species))
            
        # ---> 这是关键修改点：智能拼接路径，防止目录移动导致找不到文件 <---
        if model_name in ['PNNGS', 'ResGS', 'MLP', 'TabAttention']:
            # 不再信任 pipeline 字典里可能已经失效的绝对路径，而是强制在当前的 model_dir 下寻找同名权重
            expected_pt_name = f"{pheno}_{model_name}_weights.pt"
            dl_weight_path = os.path.join(args.model_dir, expected_pt_name)
            
            if not os.path.exists(dl_weight_path):
                raise FileNotFoundError(f"【致命错误】找不到深度学习的权重文件: {dl_weight_path}。请确保你将 .pt 文件和 .pkl 文件放在了同一个目录下！")
            
            print(f"  -> 正在加载 PyTorch 权重矩阵执行深度推断...")
            res_pred = inference_dl_model(X_fused_unknown_s, model_name, dl_weight_path)
        else:
            res_pred = functional_model.predict(X_fused_unknown_s)
            
        final_pred = np.maximum(base_pred + res_pred, 0)
        
        temp_df = pd.DataFrame({pheno: final_pred}, index=curr_predict_species)
        final_predictions_df = final_predictions_df.join(temp_df, how='left')
        print(f"  [✔] {pheno} 预测完成！")

    out_csv = os.path.join(args.outdir, "Unknown_Species_Virulence_Predictions.csv")
    final_predictions_df.to_csv(out_csv)
    print(f"\n[🎉] 全组织端到端表型推断完成！高通量预测结果已保存至: {out_csv}")

if __name__ == '__main__':
    main()