# CoCoMac 离线八桶延迟网络

2026-10-24；用户选择 n0.25，分支 mm-optimizing。

## 数据契约

- 输入 stabilized_model/k0.39/n0.25 的已缩放参数，沿用原连接采样器、种子
  20260926 和有符号 float32 权重，保留重复边与自连接。
- 延迟读取连接投射/投射延迟/delay_dt1ms_data.npz 的完整
  population_delay_ms；按 labels 对齐，禁止用其旧 connected_mask 删除边。
- 固定 dt=1 ms，将 >=8 ms 合并为 8 ms，总传输 1…8 ms 对应 kernel 0…7。
- 输出 stabilized_model/delayed/n0.25，逐行按 (delay, target) 排序。
- 文件：data.npy float32[nnz]、indices.npy int32[nnz]、indptr.npy
  int64[N+1]、delay_count.npy int64[N,8]、row_task_count.npy int32[N]；
  metadata.json 保存 shape、nnz、delay_steps/off、buffer_slots=8、任务容量、
  阈值128、任务大小4096、源参数/延迟身份及排序规则。保存小型分桶表方便审计。
- 不保存逐边 delay，也不保存运行时 buffer/head/workspace。

## 第一阶段：本轮离线构建

仅新增 delayed/build.py、同级 build_test.py 和 README.md，不修改运行时模块。

- 接口 build_delayed_connectome(model_dir, *, delay_file, output_dir,
  seed=20260926, cache_dir=None) -> Path。命令行默认读取 k0.39/n0.25 和
  delay_dt1ms_data.npz，输出 delayed/n0.25，缓存 delayed/_connectome_cache。
- 复用 load_model 和 build_connectome，保持原采样顺序、批量大小和符号权重；
  延迟表两个轴按唯一 labels 对齐，要求 dt=1 ms、有限正整数延迟。
- 分片 CSR 分批合并至输出 mmap，逐行按 (delay, target) 稳定排序并同步移动
  权重，保留重复边、自连接。不分配全图 edge_delays，不创建稠密邻接矩阵。
- 行长 <=128 的任务数为零，其余为每桶 ceil(count/4096) 之和。八桶固定保留。
- metadata.json 保存模型 labels 和已对齐的 population_delay_ms 八桶表、单位
  pA、采样版本/参数、源文件 SHA256、格式版本、桶分布及耗时；delay_off 记录
  其应采用的 int32/int64 dtype。不保存运行状态。
- 检查每行排序、范围、符号、计数和任务容量，以及全部 population 对预算后
  发布完成目录。临时目录同级，metadata 最后写入，失败保留临时目录；已有
  正式输出报错，不覆盖。小图测试通过后执行一次全量 n0.25 构建。

## 第二阶段：运行时接入（2026-10-24 批准实施）

1. 删除 Connectome.build_delay_tcsr_network 旧方法中的延迟采样和排序，改为模块级
   build_delay_tcsr_network(prepared_dir, model, *, dt_ms, binary_backend=None)。
   mmap 加载五个 NPY，校验格式、模型参数和配置哈希、labels、形状、dt 与调度
   参数。权重保持 float32 pA 数值，启用 x64 保持 int64 行指针和分段计数。
2. delay_main.py 新增 DelayMetadata NamedTuple：delay_steps、delay_count、
   delay_off、row_task_count、task_capacity、buffer_slots。构造函数允许
   edge_delays=None，并新增互斥的 delay_metadata 参数；保留旧逐边输入路径。
   预计算路径只复核元数据，不展开边、不采样或排序，检查计数/偏移/任务数和
   容量一致性。复用原后端准备和状态分配，大数组检查只回读标量结果。
3. trace.run 新增 delay_dir=None，CLI 新增 --delay-dir。无延迟分支保持原流程；
   延迟分支读取已构建目录并构造 DelayLIF，显式传入 dt_ms=1。继续共用 simulate、
   记录器及绘图。运行记录保存延迟定义、数据目录/模型身份、实际后端、图采样
   种子与动力学运行种子。延迟模式不调用 build_connectome，不另建 run_delay.py。
4. cuSPARSE 初始化 split 一次，CUDA/JAX 直接使用原 CSR，底层核不改。
   实施中发现 _DelayConnection._inputs 被捕获成 JIT 常量；全量 CSR 约 19 GB，
   因此 lif_delay.py 增加必要的小改动：将只读输入注册为 BrainState State，计算
   时读取 .value，使其作为动态参数传入。动力学与队列状态更新顺序保持不变。
   增加 JAXPR 回归测试，确保权重/索引不嵌入编译常量。
5. 相关测试整理为模块同级 *_test.py，迁移旧接口调用，不保留同名副本。覆盖
   两种构造路径数值和状态一致、非法元数据/来源、无在线采样排序、三后端小图、
   连续/分段 simulate、CLI 与无延迟回归。新增修改路径 >90%，mypy 无错误，
   JAX 0.8.0 验证 CPU；当前环境完成全量 n0.25、cuda_raw、100 步（100 ms）
   短程运行，burn_in=0、关闭绘图，检查输出和运行状态而不宣称生物动力学验证。

## 验证

小图独立参考覆盖 label 重排、1/7/8/18 ms、旧掩码差异、空行/空图、重复边、
符号权重、排序、任务128/129与4096/4097、可复现性、非法输入、文件重载与失败
发布；实现和测试通过 mypy，有效路径覆盖率>90%。离线程序仅依赖 NumPy 和
现有采样器，不导入 JAX/CUDA，同时在现有与 JAX 0.8.0 环境运行小图验证。
全量 n0.25 构建确认 1,032,387 行、2,396,907,279 条边，记录实占空间、桶分布
和耗时。本轮不执行 GPU 加载或模拟；三后端验证属于第二阶段。

## 第一阶段验收记录（2026-10-24）

- build.py 及同级测试已实现；两套环境各 31 项测试通过，覆盖率 95%，mypy 无错误。
- n0.25 全量输出已完成：1,032,387 行、2,396,907,279 边，任务容量 3,652,449。
- NPY 合计 19,253,720,292 字节，总耗时 616.48 秒；八桶连接总数与原始
  population 预算一致，重新加载后的逐行计数、任务数及源文件哈希验证通过。
- 桶分布和阶段耗时记录于 delayed/README.md、n0.25/metadata.json，完整构建
  日志为 delayed/build_n0.25.log。运行时模块未修改。

## 第二阶段验收记录（2026-10-24）

- 模块级加载函数、DelayMetadata 直接构造和 trace --delay-dir 已接通，旧在线延迟
  采样/排序方法已删除。原无延迟运行及逐边 delay_tcsr 输入保留；三个测试模块
  已迁移到实现同级，旧位置无同名副本。
- 当前 JAX 0.11.0 GPU 套件 185 项通过，包含 jax/cuda_raw/cusparse 数值验证；
  CPU 主套件 162 项通过、20 项 GPU 跳过，新增的三个不完整头校验也通过。
  隔离 JAX 0.8.0 CPU 主套件 162 项通过、20 跳过，最终加载器套件 32 项通过、
  4 跳过。七个实现/测试文件通过 mypy。
- 有效行覆盖率：delay_main 97.66%、connectome 92.13%、trace 100%、lif_delay 100%。
  connectome 新加载函数全部覆盖，未覆盖行属于原采样器的错误处理等路径。
- 全量 n0.25 命令行运行成功：cuda_raw，dt=1 ms，100 步，burn_in=0，关闭绘图。
  加载/编译/模拟/记录共 10.78 秒，1,413,915 次 spike，记录完成步数 100。
  结果为 cocomac/runs/delay_n0.25_smoke_20261024，旁边同名 .log 保存资源计时。
- 独立 100 步 simulate 状态验收：head=4、status=[55749,0]、电压/电流/buffer
  有限且已消费槽为零；报告保存在结果目录 state_check.json。此项为独立验收
  运行，未将状态推断为前一次命令行进程的直接快照。
- 回归修复：先拒绝未开启 x64 的 int64 索引，再执行设备校验；防止检查本身
  触发类型收窄。缺少任务容量或图种子的元数据在上传前统一拒绝。只读 CSR
  输入的 BrainState 注册避免全图数据进入编译常量，并有 JAXPR 测试覆盖。

## 默认运行时长调整（2026-10-24）

- trace.run 和命令行默认总时长统一为 20000 ms，burn-in 为前 10000 ms。
- 一次连续模拟全部 20000 ms，预热结束时不重置神经元、buffer 或 head。
  保存全部 spike，平均发放率仅统计后 10000 ms；不增加独立的编译预热。
- 显式 duration_ms/burn_in_ms 参数及对应 CLI 选项仍可覆盖默认值。
- 验证 Python/CLI 默认值一致，并运行已有记录器边界、参数校验和运行分支测试。

## 完整 trace 运行与记录（2026-10-24）

- 使用 k0.39/n0.25 参数和 delayed/n0.25 离线 CSR，在空闲 GPU 0 上运行
  cuda_raw、dt=1 ms、总时长 20000 ms、burn-in=10000 ms、seed=20260926。
- 保留全部 20000 步 spike、配置、统计和日志；记录墙钟耗时、峰值主机内存、
  软件版本及文件规模，核对分片连续性、事件范围和预热后统计一致性。
- trace CLI 使用 --skip-plot，随后从保存的统计和 spike 单独出图：全程发放率
  标注前 10 秒预热，脑区热图和均值使用步 10001…20000，V1/V2/FEF raster
  使用 13000…13500 ms、每个 population 固定抽取 3% 神经元。原七面板绘图
  的分析窗口和横轴刻度均固定于旧时间范围，因此不用于此次报告，也不修改
  公共绘图模块。将分析脚本一并保存于本次运行目录。
- 在独立运行目录写报告和复现命令。CLI 耗时包含加载、编译、计算和记录，
  与后续绘图耗时分别报告，不作为纯 CUDA 内核性能。

### 完整运行验收

- 输出 cocomac/runs/delay_n0.25_20s_burn10s_20261024，README.md 保存完整报告。
- 20000 步、20 个记录分片，真实 cuda_raw 运行退出码 0，墙钟 134.40 秒，
  峰值主机 RSS 26.98 GiB。simulation 记录 13,426,440,182 字节。
- 全程 1,115,481,066 次 spike，预热后 578,433,696 次；全网络加权平均
  发放率 56.028766 Hz。部分 population 发放率很高，V1/5E 为 324.686 Hz。
- 独立逐片重算全部逐毫秒 population 计数，与统计文件完全相同；分片、
  事件范围和预热后的均值校验均通过。校验及独立图表共 17.96 秒。
- 保存 PNG/SVG 图、脑区/population CSV、环境和源代码哈希、原始日志、
  资源计时及可重跑分析脚本。本次没有保存或读取最终神经元/队列状态快照。
