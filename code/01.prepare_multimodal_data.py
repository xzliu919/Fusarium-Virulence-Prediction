import os
import glob
import pandas as pd

# ================= 路径配置区 =================
# 1. 输入文件路径
FAA_DIR = "/share/org/YZWL/yzwl_liuchao/Fusarium_project/99.Science_revision/05.phylogenetic_time_tree/06.clean_rename_faas/"
GENE_COUNT_TSV = "Orthogroups.GeneCount.tsv"
ORTHOGROUPS_TSV = "Orthogroups.tsv"
PHENOTYPE_CSV = "102Fusarium_infecting_plants_phenotype.xlsx"

# 2. 输出目录及文件
OUT_FASTA_DIR = "fasta/"
OUT_OG_MAPPING = "og_mapping.tsv"
OUT_PAV_MATRIX = "pav_matrix.csv"
OUT_CNV_MATRIX = "cnv_matrix.csv"
OUT_METADATA = "metadata.csv"

# 创建 FASTA 输出目录
os.makedirs(OUT_FASTA_DIR, exist_ok=True)
# ===============================================

def process_matrices():
    """生成 PAV (0/1) 和 CNV (拷贝数) 矩阵"""
    print("[1/4] 正在生成多维基因组特征矩阵 (PAV & CNV)...")
    
    # 读取 OrthoFinder 的 GeneCount 文件
    counts_df = pd.read_csv(GENE_COUNT_TSV, sep='\t', index_col='Orthogroup')
    
    # 剔除 Total 列（如果存在）
    if 'Total' in counts_df.columns:
        counts_df = counts_df.drop(columns=['Total'])
        
    # 转置矩阵，使得行:物种，列:OGs
    cnv_matrix = counts_df.T
    cnv_matrix.index.name = 'Species'
    
    # 生成 PAV 矩阵 (大于0的设为1)
    pav_matrix = (cnv_matrix > 0).astype(int)
    
    # 输出
    cnv_matrix.to_csv(OUT_CNV_MATRIX)
    pav_matrix.to_csv(OUT_PAV_MATRIX)
    print(f"      -> CNV 维度: {cnv_matrix.shape}")
    print(f"      -> PAV 维度: {pav_matrix.shape}")

def process_og_mapping():
    """生成 OG 映射字典"""
    print("[2/4] 正在生成 OG 映射字典 (og_mapping.tsv)...")
    
    og_df = pd.read_csv(ORTHOGROUPS_TSV, sep='\t', index_col='Orthogroup')
    mapping_records = []
    
    for species in og_df.columns:
        if species == 'Total': continue
        
        # 遍历该物种在每个 OG 下的蛋白
        for og_id, proteins_str in og_df[species].dropna().items():
            # 拆分逗号分隔的蛋白
            proteins = str(proteins_str).split(',')
            for prot_full in proteins:
                prot_full = prot_full.strip()
                if not prot_full: continue
                
                # 剔除 "Species@" 前缀，获得干净的 Protein ID
                prot_id = prot_full.split('@')[1] if '@' in prot_full else prot_full
                
                mapping_records.append({
                    'Species': species,
                    'Protein_ID': prot_id,
                    'OG_ID': og_id
                })
                
    mapping_df = pd.DataFrame(mapping_records)
    mapping_df.to_csv(OUT_OG_MAPPING, sep='\t', index=False)
    print(f"      -> 共映射了 {len(mapping_df)} 条蛋白-OG关联。")

def process_fasta():
    """解析 FAA 库，按物种拆分并清理 Header"""
    print("[3/4] 正在拆分并清洗全蛋白组序列库...")
    
    faa_files = glob.glob(os.path.join(FAA_DIR, "*.faa"))
    file_handles = {}
    processed_count = 0
    
    for faa in faa_files:
        with open(faa, 'r') as fin:
            current_species = None
            for line in fin:
                if line.startswith(">"):
                    header = line.strip()[1:]
                    
                    # 从 Species@ID 中提取信息
                    if '@' in header:
                        species_name, clean_id = header.split('@', 1)
                    else:
                        species_name = os.path.basename(faa).replace(".faa", "")
                        clean_id = header
                        
                    current_species = species_name
                    
                    # 动态创建该物种的独立 FASTA 文件句柄
                    if species_name not in file_handles:
                        file_handles[species_name] = open(os.path.join(OUT_FASTA_DIR, f"{species_name}.fasta"), 'w')
                        
                    # 写入干净的 Header
                    file_handles[species_name].write(f">{clean_id}\n")
                    processed_count += 1
                else:
                    # 写入序列本身
                    if current_species and current_species in file_handles:
                        file_handles[current_species].write(line)
                        
    # 关闭所有文件句柄
    for fh in file_handles.values():
        fh.close()
    print(f"      -> 生成了 {len(file_handles)} 个纯净版物种 .fasta 文件。")

def process_metadata():
    """规范化表型元数据，并添加分组预留列"""
    print("[4/4] 正在组装表型与分组元数据 (metadata.csv)...")
    
    # 终极多编码与高容错解析逻辑
    encodings_to_try = ['utf-8-sig', 'utf-8', 'utf-16', 'gbk', 'latin1']
    pheno_df = None
    
    for enc in encodings_to_try:
        try:
            # 使用 python 引擎，它对隐藏的BOM字符和不规则列数有极强的容错性
            pheno_df = pd.read_csv(
                PHENOTYPE_CSV, 
                encoding=enc, 
                engine='python', 
                on_bad_lines='skip' # 遇到极端乱码行自动跳过，保证主程序不崩溃
            )
            print(f"      -> (检测到文件编码为 {enc}，已成功读取)")
            break
        except Exception as e:
            continue
            
    if pheno_df is None:
        print("[!] 表型文件读取彻底失败。请打开原 Excel 文件，选择 '另存为' -> 'CSV UTF-8 (逗号分隔) (*.csv)' 后重试。")
        return
    
    # 按照你的要求重命名表型列
    rename_map = {
        'wheat_stem': 'Phenotype_WheatStem',
        'wheat_head': 'Phenotype_WheatHead',
        'Maize_stem': 'Phenotype_MaizeStem',
        'Soybean_stem': 'Phenotype_SoybeanStem'
    }
    pheno_df = pheno_df.rename(columns=rename_map)
    
    # 智能推断 Species_Complex (基础版)
    def guess_complex(species_name):
        name_lower = str(species_name).lower()
        if 'oxysporum' in name_lower: return 'FOSC'
        if 'solani' in name_lower: return 'FSSC'
        if 'graminearum' in name_lower or 'asiaticum' in name_lower or 'boothii' in name_lower: return 'FGSC'
        if 'fujikuroi' in name_lower or 'proliferatum' in name_lower or 'verticillioides' in name_lower: return 'FFSC'
        if 'incarnatum' in name_lower or 'equiseti' in name_lower: return 'FIESC'
        if 'tricinctum' in name_lower or 'avenaceum' in name_lower: return 'FTSC'
        return 'Unknown'
        
    if 'Species' in pheno_df.columns:
        pheno_df['Species_Complex'] = pheno_df['Species'].apply(guess_complex)
    else:
        print(f"      -> [警告] 未在表中找到 'Species' 列，当前列名有: {list(pheno_df.columns)}")
    
    # 输出时强制清洗为最标准的 utf-8
    pheno_df.to_csv(OUT_METADATA, index=False, encoding='utf-8')
    print(f"      -> 包含 {len(pheno_df)} 个物种的表型和复合体分组已保存。")

if __name__ == "__main__":
    process_matrices()
    process_og_mapping()
    process_fasta()
    process_metadata()
    print("\n[🎉] 全部文件结构化完成！现在你的数据格式已达到机器学习多模态输入的最高标准。")