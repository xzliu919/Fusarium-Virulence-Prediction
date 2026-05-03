# 请确保安装了以下包
# install.packages(c("ggplot2", "dplyr", "tidyr", "patchwork", "ggsci", "readr", "extrafont"))

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggsci)
library(readr)

# 1. 设定输入路径与输出路径
input_dir <- "." # 请根据实际情况修改
out_pdf <- "Fusarium_Comprehensive_Model_Evaluation_A4.pdf"

cv_data <- read_csv(file.path(input_dir, "cv_fold_metrics.csv"), show_col_types = FALSE)
ho_data <- read_csv(file.path(input_dir, "holdout_metrics.csv"), show_col_types = FALSE)

# 2. 转换为长表 (Long format)
cv_long <- cv_data %>%
  pivot_longer(cols = c(R2, Pearson_r, Spearman_rho, RMSE, MAE), 
               names_to = "Metric", values_to = "Value")

ho_long <- ho_data %>%
  pivot_longer(cols = c(R2, Pearson_r, Spearman_rho, RMSE, MAE), 
               names_to = "Metric", values_to = "Value")

# 3. 异常值保护（将极端爆炸的负数 R2 截断在 -1）
cv_long <- cv_long %>% mutate(Value = ifelse(Metric == "R2" & Value < -1, -1, Value))
ho_long <- ho_long %>% mutate(Value = ifelse(Metric == "R2" & Value < -1, -1, Value))

# 4. 锁定分类顺序
model_levels <- c("Ridge", "SVR", "RandomForest", "GBR", "MLP", "PNNGS", "ResGS", "TabAttention")
metric_levels <- c("Pearson_r", "Spearman_rho", "R2", "RMSE", "MAE")
pheno_levels <- c("Phenotype_WheatHead", "Phenotype_WheatStem", "Phenotype_MaizeStem", "Phenotype_SoybeanStem")

cv_long$Model <- factor(cv_long$Model, levels = model_levels)
cv_long$Metric <- factor(cv_long$Metric, levels = metric_levels)
cv_long$Phenotype <- factor(cv_long$Phenotype, levels = pheno_levels)

ho_long$Model <- factor(ho_long$Model, levels = model_levels)
ho_long$Metric <- factor(ho_long$Metric, levels = metric_levels)
ho_long$Phenotype <- factor(ho_long$Phenotype, levels = pheno_levels)

# 5. 公共主题设置 (Arail/sans, 6pt 极限排版)
# ggplot2 的 base_size 是磅(pt)，设为 6 就是你要求的 6 号字
pub_theme <- theme_bw(base_size = 6, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 8, hjust = 0), # 标题稍微大一点点（8pt）以示区分
    plot.tag = element_text(face = "bold", size = 10),            # A, B 标签
    strip.background = element_rect(fill = "#f1f3f5", color = "black", linewidth = 0.3),
    strip.text = element_text(face = "bold", size = 6),
    panel.border = element_rect(color = "black", linewidth = 0.3),
    axis.line = element_line(linewidth = 0.3),
    axis.ticks = element_line(linewidth = 0.3),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", color="black", size = 5), # X轴文字极小，防重叠
    axis.text.y = element_text(color="black", size = 5),
    axis.title = element_text(face = "bold", size = 6),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(face = "bold", size = 6),
    legend.key.size = unit(0.3, "cm"),
    legend.margin = margin(0,0,0,0)
  )

# ==========================================
# 绘图 A: 10-Fold CV 散点箱线图
# ==========================================
p_cv <- ggplot(cv_long, aes(x = Model, y = Value, fill = Model)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA, linewidth = 0.2) + 
  geom_jitter(width = 0.2, size = 0.3, alpha = 0.8, color = "black", shape = 21, stroke = 0.1) + 
  facet_grid(Metric ~ Phenotype, scales = "free_y") +
  scale_fill_npg() + 
  labs(
    title = "Internal Learning Stability: 10-Fold Cross-Validation Metrics",
    x = "", y = "Metric Value"
  ) +
  pub_theme +
  theme(
    axis.text.x = element_blank(), 
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank()
  )

# ==========================================
# 绘图 B: Holdout 盲测带有数值标签的柱状图
# ==========================================
p_ho <- ggplot(ho_long, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black", width = 0.7, linewidth = 0.2) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.3) + 
  # geom_text 的 size 参数单位是 mm (约 1mm = 2.8pt)。要想接近 6pt (约 2.1mm)，设为 1.8 比较合适
  geom_text(aes(label = sprintf("%.2f", Value),
                vjust = ifelse(Value >= 0, -0.4, 1.4)), 
            position = position_dodge(width = 0.7), 
            size = 1.6, color = "black", family = "sans") +
  facet_grid(Metric ~ Phenotype, scales = "free_y") +
  scale_fill_npg() +
  labs(
    title = "External Generalization: Holdout Test on Reserved Distant Species",
    x = "Machine Learning & Deep Learning Architectures", y = "Metric Value"
  ) +
  pub_theme +
  theme(legend.position = "none") 

# ==========================================
# 使用 Patchwork 组合 A 与 B
# ==========================================
final_plot <- p_cv / p_ho + 
  plot_annotation(
    tag_levels = 'A', 
    theme = theme(plot.tag = element_text(size = 12, face = "bold", family = "sans"))
  ) +
  plot_layout(heights = c(1, 1)) # 上下等高

# ==========================================
# 导出 PDF (严格 A4 竖向版式)
# ==========================================
# A4 尺寸 (英寸)：宽 8.27, 高 11.69
# 注意：由于使用了 6pt 字体，必须输出在这个绝对物理尺寸下，文字和图表的比例才是最终出版要求的效果
ggsave(out_pdf, plot = final_plot, width = 8.27, height = 11.69, units = "in", device = "pdf")

cat(paste0("Success! A4 sized Publication-ready PDF generated: ", out_pdf, "\n"))