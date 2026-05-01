# 运行前请确保 pip install biopython
from Bio import Phylo
import pandas as pd
import numpy as np

tree = Phylo.read("102fusarium_phylogeny.nwk", "newick")
terminals = tree.get_terminals()
names = [t.name for t in terminals]

dist_matrix = np.zeros((len(names), len(names)))
for i in range(len(names)):
    for j in range(len(names)):
        if i == j: continue
        # 计算两个物种在进化树上的枝长距离
        dist_matrix[i, j] = tree.distance(terminals[i], terminals[j])

df = pd.DataFrame(dist_matrix, index=names, columns=names)
df.to_csv("phylo_dist.csv")
print("进化距离矩阵转换完成！")