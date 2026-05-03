library(ggplot2)
library(dplyr)
library(readr)
library(purrr)

# 1. 配置
input_dir <- "."
output_file <- "Virulence_Prediction_Scatter_A4.pdf"

# 2. 读取并合并数据
scatter_files <- list.files(input_dir, pattern = "_scatter_data.csv", full.names = TRUE)
df <- map_df(scatter_files, read_csv)

# 3. 准备标签映射与因子排序
df <- df %>%
  mutate(
    Tissue = case_when(
      Phenotype == "Phenotype_WheatHead" ~ "Wheat head",
      Phenotype == "Phenotype_WheatStem" ~ "Wheat stem",
      Phenotype == "Phenotype_MaizeStem" ~ "Maize stem",
      Phenotype == "Phenotype_SoybeanStem" ~ "Soybean stem"
    ),
    # 构造图例文字：组织名 \n 模型名 R = 数值
    Legend_Label = paste0(Tissue, "\n", Model, " R = ", round(R, 2))
  )

# 【关键点】锁定图例顺序，确保颜色和形状能精准对应
# 按照 WheatHead, WheatStem, MaizeStem, SoybeanStem 的顺序排列
legend_levels <- df %>%
  arrange(factor(Phenotype, levels = c("Phenotype_WheatHead", "Phenotype_WheatStem", "Phenotype_MaizeStem", "Phenotype_SoybeanStem"))) %>%
  pull(Legend_Label) %>%
  unique()

df$Legend_Label <- factor(df$Legend_Label, levels = legend_levels)

# 4. 颜色与形状配置
# 颜色：深蓝 (#1a2a6c), 紫红 (#b21f66), 墨绿 (#4a7c59), 深青 (#004b57)
my_colors <- c("#1a2a6c", "#b21f66", "#4a7c59", "#004b57")
my_shapes <- c(17, 16, 15, 3)

# 5. 绘图
p <- ggplot(df, aes(x = Predicted_Value, y = True_Value, color = Legend_Label, shape = Legend_Label)) +
  geom_point(size = 3, stroke = 1) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  scale_color_manual(values = my_colors) +
  scale_shape_manual(values = my_shapes) +
  labs(
    x = "Predicted lesion length (cm)",
    y = "True lesion length (cm)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    # 【修复点】使用 "sans" 代替 "Arial"，确保在 Linux 服务器上不报错
    text = element_text(family = "sans"), 
    panel.border = element_rect(color = "black", linewidth = 1),
    panel.grid = element_blank(),
    axis.ticks = element_line(color = "black"),
    legend.position = c(0.95, 0.05),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3),
    legend.title = element_blank(),
    legend.text = element_text(size = 8, lineheight = 0.8),
    legend.key.height = unit(1, "cm")
  )

# 6. 保存 (指定使用 cairo_pdf 往往能提供更好的字体嵌入支持，如果系统支持的话)
# 如果依然报错，可以尝试将 device = "pdf" 改为 device = cairo_pdf
ggsave(output_file, plot = p, width = 5, height = 6, units = "in")

print(paste("Plot saved to:", output_file))