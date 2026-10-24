# SDDMV benchmark i64 索引支持

实现已确认的 `brainevent/_tcsr/dev/sddmv/benchmark/dev/md/large_nnz_plan.md`，仅修改 benchmark 配套机制，算子 CUDA 源码保持不变。

- 全局 nnz 使用 int64；TCSR 偏移保持 int64，hybrid 在 nnz 超过 INT32_MAX-4096 时使用 int64 行指针及任务边界，cuSPARSE 在超过 INT32_MAX 时使用 64/64 索引。提供小用例强制 i64 的测试选项。数值保持 FP32。
- 按完整行分块生成/排序，默认块 nnz 上限 2^30；保持既有随机数据、重复列和排序顺序。准备块不改变每步完整计算的计时语义。
- 按方案与事件类型估算阶段峰值，筛选条件为已知缓冲区严格小于 96 GiB；补充分配、实际 workspace 和可用显存检查。
- GPU 全量正确性检查及 L1 分块归约在计时外进行，错误计数与位置支持 64 位；保留 prezeroed 多步覆写语义。
- 验收包括普通回归、强制 i64/分块、超过 2^31 与 2^32 的真实用例、高位故障注入和近 80 GiB cuSPARSE 用例。已知算子 i32 边界问题保持独立待办。
