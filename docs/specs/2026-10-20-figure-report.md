# 10.20 图文报告

- 在当前 `mm-optimizing` 分支的 `brainevent/_tcsr/dev/docs/10.20/report.md` 创建报告。
- 报告顶部增加 TODO List，按用户顺序列出七项任务、`/`、`√`、`×` 状态及进展说明；将“更搞笑的算子”按上下文修正为“更高效的算子”。
- 报告正文分为 `mm`、`sddmv`、`sddmm`、`delay_mv`、`mv 双端`、`视觉皮层模型` 六节，包含用户指定的十五张 SVG、简短配文及后续要求补充的 MM、SDDMV 调用过程说明。
- 相邻章节之间用 `<br><br>` 增加留白，避免普通 Markdown 空行在渲染时折叠。
- `mm`：Tile SpMM 示意图后集中排列三张结果图，batchmajor/tile 热图在上，batchmajor/cuSPARSE 与 tile/cuSPARSE 曲线图在下并排。
- MM 示意图配文后、结果图前使用原回复中的 TCSR、cuSPARSE 两个 `text` 代码块，保留逐行换行、箭头和计时桩内缩进；明确 TCSR 辅助状态数组在桩内清零、共享缓冲在 Kernel 4 内清零并覆盖全局输出，cuSPARSE 全局输出在桩外清零；保留逐步记录事件、轮末同步求和的顺序。此前将“两段文本”误解为连续段落，后续以用户指定的原回复格式为准。
- `sddmv`：Hybrid 示意图后依次放 Hybrid prezeroed 与原基线 prezeroed 的结果图，上下排列；注明二者均为加速比乘以事件密度的归一化百分比。
- SDDMV 示意图配文后、结果图前补充三个 `text` 流程块：Hybrid pre / SDDMV pre、Hybrid / SDDMV（不带 pre）、cuSPARSE。每个窗口最多 32 步，输入生成及 bool 转 float 在桩外，窗口结束同步取时；pre 仅初始化时清零 dW，非 pre 每步在桩内清零 dW，辅助状态每步在桩内清零，cuSPARSE 在桩内逐步更新输入指针并以 beta = 0 调用 cusparseSDDMM。此前误将两种 pre 方案拆成两类，后续按用户指定的是否在桩内清零全局输出分组。
- SDDMV 的 pre 与非 pre 流程在初始化行下补充输入池形态：GPU 连续缓冲 `events[32, n]`、`ct[32, n]`，容纳 32 步、每步 n 个神经元的输入，每个窗口重新填充。
- `sddmm`、`delay_mv`：分别放二值 SDDMM、脑区投射延迟示意图。
- `mv 双端`：先放双端任务列表示意图，再放 float32、Gini = 0.0 下相对原 Hybrid 的加速比热图；加速比为原 Hybrid 耗时 / 双端耗时。
- `视觉皮层模型`：按 `dt1ms`、`dt1ms0.3We3taue` 两组排列，各组先放 `fig3_no_delay.svg`，再放 `raster_5000_proportional.svg`，使用报告目录下 `fig/` 中的原图；第二组文件位于其 `figures/` 子目录。配文分别说明主要脑区的统计指标、所有脑区的放电情况，图号续为 12–15。
- 用相对路径引用原图及对应 TeX 图稿；示意图用一句配文概括各 kernel 的主要工作，图稿链接融入图名；结果图只说明对照方案与指标。
- 配文不展开线程组织、颜色、存储布局及模式差异；后续新增配文保持相同粒度，避免再次加入过多实现细节。
- 验收：核对图片顺序、链接可解析、配文与图稿一致及 Markdown 空白格式；不涉及代码或数值测试。
