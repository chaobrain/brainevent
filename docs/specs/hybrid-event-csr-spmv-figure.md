# Hybrid event-driven CSR transpose SpMV 图示

## 目标

在 brainevent/_tcsr/dev/Latex_alg/fig/hybrid_spmv 提供可编辑的 TikZ
源文件、渲染脚本和独立 SVG。遵守 fig/README.md，以
tile_mm_kernel4/tile_spmm_kernel4.svg 为视觉参考。

## 图中结构

1. 左侧画事件向量 x 与物理 CSR 矩阵 A (m×n)。两个活动行分别代表
   短行和长行；网格中的非零元仅示意稀疏位置，数量不按阈值比例绘制。
2. 第一轮画一个蓝框 warp，框内用多条波浪箭头代表线程。两个选中线程
   分别以虚线箭头指向短行与长行；矩阵行通过水平、竖直组成的实线折箭头
   接到 TPR 与 Task queue 左边框，箭头不得进入模块内部。
   下移 warp 和 TPR，两个 Kernel 面板顶边对齐，减少画布上方留白。
   线程组织关系由虚线表示，
   行到模块的数据路径由实线表示。任务队列显示
   task_begin 与 task_end 两个列表，各有索引 0、1 两项，按相同索引
   配对形成两个任务。
3. 第二轮保留任务对应的 CUDA block，每个 block 内用一个代表性的 warp
   蓝框替代展开的 Warp 0–7 列表；warp 框指向 task 数据区域。
   实际默认仍为 256 线程、8 个 warp，任务由 block 取得。
4. 结构之间使用水平与竖直折线；warp 内部使用参考图式的波浪箭头。
   输出重合列用上下双色三角形组成与其他标记等大的无边框方块。
5. 所有 warp 外框使用正方形；线程曲线减小横向摆幅。标准图元在
   fig/warp_block.tikz 中统一定义，主图和 fig/warp_block.svg 共用。
   标准 SVG 以正方形画布独立输出，另附源文件及渲染脚本供直接复用。

## 准确性与验收

- 图面文字只标识结构名称、维度和索引。阈值、计算行为和示意范围仅写在
  图外文档或 TeX 注释中。
- 第二轮以 CUDA block 结构呈现；TPR 只标注第一轮短行的 thread-per-row。
- 任务队列仅包含区间边界，不画成权重或列索引数组。
- LaTeX 编译与 SVG 转换成功；输出自包含、无光栅图片；人工检查预览中的
  箭头方向、标签和颜色，并验证图不裁切。
