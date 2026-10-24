# 延迟投射基准的后端选择

`benchmark_binary_csr_vecmat.py` 的每个 `CASE_LIST` 条目增加 `BACKEND_LIST`，
取值为按执行顺序排列的 `"hybrid"`、`"cusparse"` 子集。当前默认配置只运行 `hybrid`；
自定义旧配置缺少该键时也运行两者。空列表、重复项和未知名称应立即报错。

`run_all_cases` 将该列表传给 `benchmark_case`。输入矩阵仍为一次生成、一次上传，
只调用选中的原生后端，只为选中的后端写入结果和初始化失败行。
仅选 `hybrid` 时，显存预算不计入不会创建的 cuSPARSE 拆分数组。
烟雾测试配置显式列出两个后端。CSV 与汇总文件的字段保持不变。

测试覆盖单后端选择、列表顺序、失败行范围、非法列表，以及旧配置的默认行为。

脚本开头显式声明默认 `GPU_ID`。直接运行脚本时，在导入 NumPy 和项目的 GPU
相关模块之前，将命令行 `--gpu`（缺省为 `GPU_ID`）写入
`CUDA_VISIBLE_DEVICES`。作为模块导入时不改写调用进程的设备选择。
