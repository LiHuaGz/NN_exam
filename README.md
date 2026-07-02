# 神经网络期末试题

本仓库包含课程期末试题的 LaTeX 报告、Python 实验程序、实验输出结果和题目附件。

## 目录结构

```text
.
|-- README.md                         # 仓库说明
|-- LICENSE                           # 开源许可证
|-- main.tex                          # 课程报告 LaTeX 源文件
|-- main.pdf                          # 编译后的报告
|-- references.bib                    # 参考文献
|-- 试卷.pdf                          # 试题 PDF
|-- 附件.zip                          # 原始附件压缩包
|-- 附件3 复旦大学研究生课程试卷（样张）2026.docx
|-- data/
|   `-- MNIST/
|       `-- raw/                      # MNIST 原始数据
|-- python/
|   |-- Q1.py                         # 第1题数值验证
|   |-- Q1_outputs/                   # 第1题输出图表和 CSV
|   |-- Q4.py                         # 第4题泊松群体编码实验
|   |-- Q4_outputs/                   # 第4题输出图表和数据
|   |-- Q5.py                         # 第5题盲源分离实验
|   |-- Q5_outputs/                   # 第5题分离音频和指标图表
|   |-- Q6.py                         # 第6题 SGD 与自然梯度对比
|   |-- Q6_outputs/                   # 第6题训练曲线和结果表
|   |-- Q7.py                         # 第7题迷宫最短路径规划
|   |-- Q7_outputs/                   # 第7题路径规划结果图
|   `-- maze.jpg                      # 迷宫图片
`-- 附件/
    `-- 附件/
        |-- BSS/                      # 第5题混合音频
        `-- maze.jpg                  # 原始迷宫图片
```

## 说明

- `python/Q*_outputs/` 保存各题脚本运行后生成的图片、表格或音频结果。
- `data/MNIST/raw/` 保存第6题使用的 MNIST 数据文件。
- `.git/`、`.venv/`、`.matplotlib_cache/`、LaTeX 辅助文件和 Python 缓存文件不作为主要目录结构展示。
