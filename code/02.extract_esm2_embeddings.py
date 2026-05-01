import os
import glob
import h5py
import torch
import numpy as np
import argparse
import multiprocessing as mp
from math import ceil
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

def parse_args():
    parser = argparse.ArgumentParser(description="多卡并行提取 ESM-2 蛋白序列特征并构建 HDF5 数据库")
    
    # 核心路径参数
    parser.add_argument("--fasta_dir", type=str, required=True,
                        help="存放清洗后 .fasta / .faa 文件的目录路径")
    parser.add_argument("--model_path", type=str, default="/data/liuchao/01.Model/esm2_t33_650M_UR50D",
                        help="本地 ESM-2 模型的绝对路径")
    parser.add_argument("--output_h5", type=str, default="protein_embeddings.h5",
                        help="最终输出的 HDF5 数据库文件路径")
    
    # 计算与硬件参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="单卡推理的 Batch Size。A800 80G 显存对于 650M 模型可放心开到 32 或 64")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="序列最大截断长度。超长序列将被截断以防显存溢出 (OOM)")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="使用的 GPU 数量，将自动分配到 cuda:0 到 cuda:N-1")
    
    return parser.parse_args()

def mean_pooling(model_output, attention_mask):
    """
    对模型输出的最后一层隐藏状态进行 Mean Pooling，同时忽略 Padding 部分的 token。
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def worker_process(gpu_id, file_chunk, args, temp_h5_path):
    """
    单个 GPU 的工作进程：加载模型，处理分配给它的 fasta 文件，将结果写入临时的 HDF5
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] 初始化中，分配到 {len(file_chunk)} 个 Fasta 文件...")
    
    # 1. 加载本地 Tokenizer 和 模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = EsmModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval() # 设置为推理模式
    
    # 2. 打开专属的临时 HDF5 文件
    with h5py.File(temp_h5_path, 'w') as h5_file:
        for fasta_file in file_chunk:
            # 提取物种名称 (文件名去除后缀)
            species_name = os.path.splitext(os.path.basename(fasta_file))[0]
            # 在 HDF5 中为该物种创建 Group
            species_group = h5_file.require_group(species_name)
            
            # 读取 Fasta 文件
            records = list(SeqIO.parse(fasta_file, "fasta"))
            
            # 批处理循环
            for i in range(0, len(records), args.batch_size):
                batch_records = records[i : i + args.batch_size]
                
                # 提取纯净的 Protein_ID (处理可能存在的 Species@ 前缀)
                batch_ids = [rec.id.split('@')[1] if '@' in rec.id else rec.id for rec in batch_records]
                batch_seqs = [str(rec.seq) for rec in batch_records]
                
                # Tokenize 编码 (自动处理截断和补齐)
                inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=args.max_len)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # 前向传播提取特征
                    outputs = model(**inputs)
                    # Mean Pooling (基于真实序列长度，去除 Padding 的影响)
                    embeddings = mean_pooling(outputs, inputs['attention_mask'])
                    
                    # 转移回 CPU 并转换为 Numpy 格式
                    embeddings_np = embeddings.cpu().numpy()
                
                # 存入临时 HDF5 文件
                for prot_id, emb in zip(batch_ids, embeddings_np):
                    # 如果库中已有该蛋白，先删除（防重复覆盖报错）
                    if prot_id in species_group:
                        del species_group[prot_id]
                    # 写入数据结构: file["Species"]["Protein_ID"] = np.array
                    species_group.create_dataset(prot_id, data=emb, dtype=np.float32)
                    
            print(f"[GPU {gpu_id}] 已完成物种: {species_name} ({len(records)} proteins)")

    print(f"[GPU {gpu_id}] 工作完成！释放显存。")
    del model
    torch.cuda.empty_cache()

def merge_h5_files(temp_h5_files, final_h5_path):
    """
    合并所有 GPU 生成的临时 HDF5 文件为一个主文件
    """
    print(f"\n[*] 正在合并 {len(temp_h5_files)} 个临时 HDF5 文件至 {final_h5_path}...")
    with h5py.File(final_h5_path, 'w') as final_h5:
        for tmp_file in temp_h5_files:
            with h5py.File(tmp_file, 'r') as tmp_h5:
                # 遍历临时文件中的 Species
                for species in tmp_h5.keys():
                    species_group = final_h5.require_group(species)
                    # 遍历该物种下的所有 Protein_ID
                    for prot_id in tmp_h5[species].keys():
                        # 直接复制底层数据块，速度极快
                        tmp_h5.copy(tmp_h5[species][prot_id], species_group, name=prot_id)
            # 合并完一个即删除临时文件
            os.remove(tmp_file)
    print("[✔] 数据库构建成功！")

def main():
    # 设置启动方式为 spawn (解决 CUDA 和多进程结合的问题)
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    
    # 1. 扫描所有 Fasta 文件
    fasta_files = glob.glob(os.path.join(args.fasta_dir, "*.faa")) + \
                  glob.glob(os.path.join(args.fasta_dir, "*.fasta"))
    
    if not fasta_files:
        raise ValueError(f"未在 {args.fasta_dir} 找到任何 .faa 或 .fasta 文件！")
    
    print(f"[*] 共发现 {len(fasta_files)} 个序列文件，准备使用 {args.num_gpus} 张 GPU 并行提取。")
    
    # 2. 将文件列表均分为 N 份 (N = num_gpus)
    chunk_size = ceil(len(fasta_files) / args.num_gpus)
    file_chunks = [fasta_files[i : i + chunk_size] for i in range(0, len(fasta_files), chunk_size)]
    
    processes = []
    temp_h5_files = []
    
    # 3. 启动多进程
    for gpu_id in range(args.num_gpus):
        # 确保不会超出可用 GPU 数量
        if gpu_id >= len(file_chunks):
            break
            
        temp_h5 = f"temp_embeddings_gpu{gpu_id}.h5"
        temp_h5_files.append(temp_h5)
        
        p = mp.Process(target=worker_process, args=(gpu_id, file_chunks[gpu_id], args, temp_h5))
        p.start()
        processes.append(p)
        
    # 等待所有 GPU 任务结束
    for p in processes:
        p.join()
        
    # 4. 合并数据
    merge_h5_files(temp_h5_files, args.output_h5)

if __name__ == "__main__":
    main()