# Attention Sequence v1

## Scope / 范围

This document defines how Transformer attention is represented on the current
unified NPU path. Attention v1 is a software/compiler scheduled primitive
sequence. It is not a dedicated `attention_engine` RTL macro.

The goal is to make attention the main Transformer workload and PPA driver while
reusing and extending the existing primitive engines:

- matrix engine;
- accumulator file;
- vector engine;
- reduction engine;
- SFU;
- data mover and local tensor memory;
- wrapper descriptor/runtime/perf path.

中文说明：

本文档定义完整Attention如何映射到统一NPU，而不是只描述Mask。Attention
不是一个独立的大型RTL模块；Compiler和Runtime将它拆分为QK矩阵乘、
Score Scale/Mask、逐行Softmax和PV矩阵乘，并调度共享的Matrix、Vector、
Reduction、SFU和Data Mover执行。本文档同时是这些阶段之间数据格式、顺序
和模块职责的端到端设计来源。

## Mathematical Contract / 数学契约

For one attention head:

```text
Q: S x D
K: S x D
V: S x D

scores = Q * K^T
scaled_scores = scores / sqrt(D)
masked_scores = apply_mask(scaled_scores)
P = softmax(masked_scores)
O = P * V
```

Expanded per element:

```text
scores[i,j] =
  sum(k=0..D-1) Q[i,k] * K[j,k]

scaled_scores[i,j] =
  scores[i,j] / sqrt(D)

masked_scores[i,j] =
  scaled_scores[i,j]                 if position j is visible to query i
  SOFTMAX_NEG_INF                    otherwise

P[i,j] =
  exp(masked_scores[i,j]) /
  sum(t=0..S-1) exp(masked_scores[i,t])

O[i,d] =
  sum(j=0..S-1) P[i,j] * V[j,d]
```

The scale by `1 / sqrt(D)` is part of scaled dot-product attention. It prevents
the dot-product magnitude from growing with `D`, which would otherwise push
softmax toward saturated one-hot outputs and make quantized attention unstable.
For v1 hardware-facing fixed point, the compiler must lower this scale into an
explicit vector/requant operation before softmax.

Masking is part of the attention semantics, not an optimization:

- prefill self-attention with a causal decoder model must prevent query position
  `i` from attending to future keys `j > i`;
- decode attention for one newly generated token usually has no future tokens in
  the current context, but it may still need a padding/valid-length mask when a
  row contains unused cache slots or tile tail lanes;
- encoder-style bidirectional attention may use no causal mask, but can still
  use padding masks.

The first measured attention target may be unmasked `S=8,D=8` to validate QK
and softmax mechanics. Causal/padding mask support must be added before claiming
decoder-style attention semantics.

中文说明：

对单个Attention head，先计算`Q × K^T`得到Score，再除以`sqrt(D)`、应用
Mask、逐行计算Softmax，最后计算`P × V`。Mask属于Attention正确性语义：
causal模型不能看到未来token，padding和tile尾部也不能参与最大值、求和或
最终概率。未实现Mask的`S=8,D=8`只能作为硬件通路验证，不能声称已经实现
decoder Attention。

## Fixed-Point Policy / 定点数策略

Detailed fixed-point derivations, examples, and golden/RTL consistency rules are
owned by `docs/design/transformer/attention_numerical_v1.md`. This section keeps
the execution-level summary.

The mathematical attention formula uses real numbers:

```text
scores / sqrt(D)
exp(x)
1 / sum(exp(x))
P * V
```

The RTL path cannot leave these as abstract floating-point operations. Every
real-valued step must be lowered to an explicit fixed-point representation,
rounding rule, clamp rule, and testable golden model.

### Probability Q0.15 convention

`Q0.15` means a fixed-point value with zero integer magnitude bits and fifteen
fractional bits. In many DSP/NPU designs, softmax probabilities are stored as
unsigned fixed-point integers because probabilities are in `[0, 1]`.

This project uses an unsigned Q0.15-like softmax probability convention:

```text
PROB_ONE = 32767
real_probability ~= p_q15 / PROB_ONE
p_q15 range = 0..32767
```

So:

```text
0.0   -> 0
0.5   -> about 16384
1.0   -> 32767
```

This is close to common Q0.15 usage. Some industry implementations instead use
Q1.15 signed values, int8 probabilities with a tensor scale, bfloat/FP16, or
block-floating formats depending on model accuracy and hardware cost. For this
project, Q0.15 is the first reviewed softmax probability format because it is
simple, deterministic, and keeps probability precision higher than int8 before
the `P*V` stage.

### Score scale policy

The real scale is:

```text
scale_real = 1 / sqrt(D)
scaled_score_real = score_int32 * scale_real
```

This floating-point value is not stored or computed as floating point in v1.
The compiler must select one of these explicit policies and record it in
workload metadata:

| Policy | Formula | Status |
| --- | --- | --- |
| `power_of_two_shift` | `scaled = score >>> score_shift` | current bring-up compatible |
| `fixed_multiplier_shift` | `scaled = round(score * multiplier / 2^shift)` | target for better `1/sqrt(D)` approximation |
| `model_only_float_reference` | Python float reference only | allowed for accuracy studies, not RTL evidence |

For `D=8`, `1/sqrt(8) ~= 0.353553`. A plain right shift cannot represent this
exactly. A fixed multiplier can approximate it, for example:

```text
multiplier = round((1 / sqrt(8)) * 2^15) = 11585
scaled = round(score * 11585 / 2^15)
```

The exact multiplier width, rounding mode, and shift belong to the vector
requant v2 contract before RTL implementation. Until that is implemented,
measured attention QK may use a documented shift approximation, while full
attention accuracy claims must use the reviewed fixed multiplier policy.

### EXP and reciprocal policy

Softmax is implemented as:

```text
delta = masked_scaled_score - row_max
exp_input = clamp(delta, -256, 0)
e_q15 = EXP_Q0_15(exp_input / 32)
sum_q15 = sum(e_q15)
recip_q24 = floor((1 << 24) / sum_q15)
p_q15 = normalize(e_q15, recip_q24)
```

The initial target is:

- EXP: generated 257-entry LUT for integer inputs `[-256, 0]`, scale `32`;
- RECIP: integer reciprocal returning unsigned Q0.24;
- normalization: vector multiply/requant from `e_q15 * recip_q24` back to
  Q0.15.

Current RTL has only a coarse 9-segment EXP bring-up model and a simple integer
division reciprocal. That path is useful for primitive wiring tests but is not
the final attention softmax numerical contract.

### Golden/RTL consistency rule

Every attention workload must identify the numerical contract used by both
Python golden and RTL. The target path must compare stage-by-stage
intermediates, not only final output:

```text
scores -> scaled_scores -> masked_scores -> row_max -> exp_input
-> e_q15 -> sum_q15 -> recip_q24 -> P_q15 -> O
```

The same generated EXP table and constants must feed both golden and RTL. If RTL
uses a coarser approximation, the workload must be labeled as bring-up or
model-only for attention PPA.

The first executable target is fixed-point, single-head attention:

| Symbol | v1 initial value | Type |
| --- | --- | --- |
| `S` | 8 first, then 16 | sequence length |
| `D` | 8 first, then 16 | head dimension |
| `Q`, `K`, `V` | int8 | activation/KV tensor |
| `scores` | int32 | exact dot-product accumulator |
| `scaled_scores` | int32 | shifted/fixed-point score |
| `P` | unsigned Q0.15 | softmax probability matrix |
| `O` | int32 first, optional int8 after requant | attention output |

The `S=8,D=8` target maps directly onto the current `8x8x8` matrix tile. Larger
`S` or `D` values are represented as tiled primitive sequences.

中文说明：

RTL不执行抽象浮点公式。Score缩放、EXP、倒数、概率归一化和PV都必须明确
输入输出位宽、Q格式、舍入、截断和饱和规则。当前概率使用unsigned Q0.15；
所有简化近似都必须在workload和PPA中标记，不能把bring-up近似当成目标
数值实现。

## Primitive Decomposition / 原语分解

Attention decomposes into five primitive groups.

Attention由五组原语组成。Compiler决定阶段和原语顺序，硬件模块只执行各自
原语；Compute Cluster不应重新硬编码一套Attention状态机。

### 1. QK Score Matmul / QK分数矩阵乘

Formula:

```text
scores[i,j] = sum(k=0..D-1) Q[i,k] * K[j,k]
```

Hardware implication:

- matrix engine executes `Q * K^T`;
- compiler/runtime must supply `K^T` layout or a tile loader that presents K as
  `D x S`;
- accumulator stores int32 score tiles.

Current status:

- `matmul_array.sv` supports one `8x8x8` int8/int8/int32 tile;
- K-streaming accumulation supports larger K in the wrapper/core path;
- current workload generator can emit deterministic `matmul_k_stream`
  fixtures.

Needed for attention:

- document and test `K^T` tile layout for QK;
- preserve logical shape metadata in workload manifests;
- expose QK cycles/MACs separately from PV cycles/MACs in perf/PPA;
- add true skinny/GEMV handling later for decode attention.

中文说明：

QK阶段使用Matrix Engine计算`Q × K^T`并生成int32 Score。物理Matrix
Engine一次只处理`8x8x8`，较大的`D`通过K-stream在Accumulator File中
累加，较大的Query/Key维度通过多个输出tile完成。完整分块设计由
`transformer_npu_v1.md`中的“大矩阵分块执行设计”负责。

### 2. Score Scale / Score缩放

Formula:

```text
scaled_scores[i,j] = scores[i,j] / sqrt(D)
```

For initial fixed-point v1 this is represented as:

```text
scaled_scores[i,j] = requant(scores[i,j], scale = 1 / sqrt(D))
```

The simplest current-compatible approximation is a power-of-two shift:

```text
scaled_scores[i,j] = scores[i,j] >>> score_shift
```

For `D=8`, `1 / sqrt(D)` is not exactly a power-of-two. Therefore this shift is
a bring-up approximation unless the workload explicitly chooses `score_shift`
as its fixed-point numerical policy. A more faithful implementation needs
`mul_round_shift_clamp`:

```text
wide = scores[i,j] * score_scale_multiplier
scaled_scores[i,j] = round(wide) >>> score_scale_shift
```

Hardware implication:

- vector engine needs a reviewed scale/requant operation for int32 score lanes;
- current `VEC_REQUANT` can only do arithmetic shift plus clamp;
- `requant_v2` is needed when attention accuracy requires non-power-of-two
  `1/sqrt(D)` scaling.

Current status:

- current vector standalone RTL has `VEC_SCALE` and `VEC_REQUANT`;
- current `VEC_REQUANT` is `a >>> shift` with clamp;
- no scheduler-integrated vector path exists yet.

Needed for attention:

- workload metadata must state `head_dim`, `score_scale_policy`,
  `score_shift` or multiplier/shift fields;
- Python golden must use the same score scale policy as the intended RTL path;
- PPA reports must identify whether scale is exact fixed-point or a bring-up
  shift approximation.
- full attention accuracy claims require the fixed multiplier policy or another
  reviewed approximation for `1/sqrt(D)`.

中文说明：

Score Scale实现除以`sqrt(D)`。硬件实际使用定点乘法加移位近似，而不是
浮点除法；Compiler选择乘数和移位，Vector Engine执行，数值文档定义舍入
和饱和。`D=8`不能用单纯右移精确表示，因此当前固定乘数方案必须保留。

### 3. Attention Mask / Attention掩码

Design status: implemented for causal single-tile `S=8,D=8`. Descriptor/uop
specs select a packed descriptor-referenced row-mask table with no new `MASK`
uop. The Data Mover loads two words into core-local mask registers before
Scheduler launch; Vector/Reduction/normalization consume the selected row mask.
Tail rows and hardware-visible malformed-descriptor error reporting remain.

Formula:

```text
masked_scores[i,j] = mask[i,j] ? scaled_scores[i,j] : SOFTMAX_NEG_INF
```

Mask meaning:

```text
mask[i,j] = 1  means key/value position j is visible to query position i
mask[i,j] = 0  means it must not affect softmax or output O
```

Common masks:

| Mask | Condition | Why it exists |
| --- | --- | --- |
| none | all `mask[i,j]=1` | useful for first prefill bring-up and non-causal attention |
| causal | `mask[i,j]=1` only when `j <= i` | decoder prefill must not see future tokens |
| valid length / padding | mask out unused sequence or cache slots | decode/cache tiles may include invalid tail lanes |
| tile tail | mask out lanes beyond logical `S` | tiled rows may be wider than logical row length |

#### Problem and value

Masking is required for correctness, but correctness alone does not justify a
new hardware block. The architecture question is where masking should execute
so that its implementation cost is lower than the movement and synchronization
cost it removes.

For a score tile with `E` elements, a CPU-materialized mask requires, at
minimum:

```text
read E int32 scores + modify E scores + write E int32 scores + synchronize
```

For the current `8x8` tile this is `256` bytes read plus `256` bytes written
before Softmax. The current NPU Scale/Mask stage already reads and rewrites the
score tile for scaling, so applying a lane predicate in that pass avoids a
separate CPU read/modify/write pass. It also establishes the mask semantics
needed by future on-chip-resident or fused Attention, where a CPU pass would
force an otherwise unnecessary SRAM boundary.

Mask support does not automatically reduce current `8x8` QK Matrix cycles:
the current Matrix engine still computes the full tile. For future multi-tile
causal Attention, the Compiler can skip completely invisible future-key tiles;
that can reduce Matrix work and movement in addition to Softmax work.

#### Industry patterns

Industry implementations generally separate mask *description* from mask
*consumption*:

- software/API describes causal, padding/sequence-length, sliding-window,
  block, or arbitrary mask semantics;
- the selected Attention kernel applies those semantics while scores are
  resident, instead of requiring a CPU pass over the complete score matrix;
- regular masks use compact metadata or index predicates, while arbitrary
  masks may require a mask tensor and additional movement.

NVIDIA cuDNN SDPA is one concrete example: its Attention API accepts causal
diagonal bounds, per-batch query/key sequence lengths for padding, block masks,
and additive bias masks. cuDNN states that padding tokens are automatically
masked during Attention computation. FlashAttention provides the architectural
motivation: reducing reads and writes between memory levels is central to
Attention performance, so materializing and revisiting the complete score
matrix is undesirable.

References:

- NVIDIA cuDNN Attention:
  `https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html`
- FlashAttention, IO-aware exact Attention:
  `https://arxiv.org/abs/2205.14135`
- FlashAttention-2, work partitioning and reduced communication:
  `https://arxiv.org/abs/2307.08691`

These references demonstrate the software/hardware split and IO motivation.
They do not imply that the current NPU should implement a dedicated fused
Attention macro.

#### Alternatives considered

| Option | Software responsibility | Hardware responsibility | Benefits | Costs / limitations |
| --- | --- | --- | --- | --- |
| A. CPU materializes masked scores | read score tile, apply all mask rules, write `SOFTMAX_NEG_INF` | execute ordinary unmasked Softmax | no new mask RTL; useful fallback and reference | extra CPU work, full score read/write, synchronization, breaks future score residency/fusion |
| B. Compiler emits dense mask tensor | allocate and move one mask value/bit per score | load mask tensor and gate lanes | supports arbitrary masks | additional storage, movement, ports, and descriptor complexity; excessive for regular causal/tail cases |
| C. Compiler emits compact row masks; existing engines gate lanes | compose logical policy into row-valid metadata | Scale/Mask selects sentinel; Reduction excludes invalid lanes; Softmax writes zero probability | removes CPU score pass; small metadata; supports causal/padding/tail; compatible with future residency | adds lane gating, mask transport, tests, and all-invalid-row handling |
| D. Hardware generates all mask rules | provide policy and shape only | generate causal/padding/tail/arbitrary behavior | compact commands for supported rules | more control complexity; risks hard-coding high-level policy; arbitrary masks still need data |
| E. Dedicated Mask engine | issue a separate mask operation | separate datapath/module rewrites tile | modular accounting | duplicates a tile pass and storage ports; no clear benefit over Vector/Reduction integration |

#### Accepted v1 architecture direction

中文说明：

当前方案不会让Compiler额外生成一条`MASK`指令。Mask是一个tile级共享属性，
如果每个row单独执行`MASK`指令，会增加程序长度以及Scheduler握手开销，但
不会增加有效计算。Compiler只生成两个32-bit mask word；Scale/Mask与
Softmax descriptor共同引用它们。现有row-indexed指令根据row编号自动选择
对应的8-bit mask。

Select **Option C**, with limited rule generation from Option D:

1. Compiler owns logical policy composition and emits:

   ```text
   valid_query_mask
   valid_lane_mask[row]
   ```

2. Runtime transports a packed row-mask table through descriptor `input1`.
   It must not infer validity from tensor values.
3. Scale/Mask consumes each row mask while it already performs fixed score
   scaling:

   ```text
   scaled = round(score * multiplier / 2^shift)
   masked_score = valid_lane ? scaled : SOFTMAX_NEG_INF
   ```

4. Reduction receives the same row mask and excludes invalid lanes from
   `REDUCE_MAX` and `REDUCE_SUM`.
5. Softmax normalization forces invalid output lanes to zero.
6. No dedicated Mask engine is added.
7. Dense arbitrary masks remain deferred. The first executable policies are
   causal, padding/valid-length, and tile-tail.

#### What "fused into Score Scale" means

The fusion is one Vector Engine read/compute/write pass, not a new combined
ISA instruction and not Wrapper-side operator fusion.

Without fusion:

```text
pass 1: read score tile -> scale every score -> write scaled tile
pass 2: read scaled tile -> replace invalid lanes -> write masked tile
```

Selected fused pass:

```text
read one score row and its row mask
  -> calculate eight fixed-point scaled results
  -> eight per-lane writeback selects
       valid lane   -> scaled result
       invalid lane -> SOFTMAX_NEG_INF
  -> write one masked/scaled row
```

The existing Scale/Mask program still issues one `VSCALE_FIXED row_index` uop
per row. `row_index` selects both the score row and
`local_row_mask_table[row_index]`. Compute-cluster routing presents the score
row, selected mask, multiplier, shift, and canonical fill value to Vector
Engine together.

```text
VSCALE_FIXED row=2
  score_row       = score_tile[2]
  lane_mask       = local_row_mask_table[2]
  multiplier      = SCORE_SCALE_MULTIPLIER
  shift           = SCORE_SCALE_SHIFT
  invalid_fill    = SOFTMAX_NEG_INF
```

Vector Engine behavior:

```text
for lane in 0..7 in the same vector transaction:
  scaled = round_fixed(score_row[lane], multiplier, shift)
  output[lane] = lane_mask[lane] ? scaled : invalid_fill
```

The current RTL cannot be used unchanged: its generic `valid_mask=0` behavior
writes zero. Zero is a legal score and would affect Softmax. The masked
`VSCALE_FIXED` implementation must instead select `SOFTMAX_NEG_INF` for an
invalid lane. This is an operation-specific inactive-lane result, while other
Vector operations retain their reviewed inactive-lane behavior.

Concrete causal-row example, using a simplified scale of one half only to make
the dataflow easy to read:

```text
query row             = 2
causal row mask       = 0b0000_0111
raw scores            = [16, 12, 8, 4, 0, -4, -8, -12]
scaled values         = [ 8,  6, 4, 2, 0, -2, -4,  -6]
fused Scale/Mask out  = [ 8,  6, 4, N, N,  N,  N,   N]
N                     = SOFTMAX_NEG_INF
```

Only one row read and one row write occur. The mask does not add a second
Vector uop or a second score-tile pass.

Descriptor-level execution sequence:

```text
ATTENTION_SCALE_MASK_V1 descriptor
  -> Data Mover loads score tile
  -> Data Mover loads/unpacks two row-mask words
  -> Uop Scheduler issues VSCALE_FIXED row 0..7
  -> each Vector transaction scales and mask-selects one row
  -> Data Mover stores the scaled/masked score tile

ATTENTION_SOFTMAX_V1 descriptor references the same row-mask table
  -> Reduction excludes invalid lanes from max and sum
  -> SFU work is consumed only according to the reviewed mask behavior
  -> Normalization forces invalid probabilities to zero
```

Writing the sentinel during Scale/Mask and carrying the same validity mask
through Softmax serve different purposes:

- the sentinel makes the stored/intermediate masked-score tile explicit and
  prevents ordinary downstream vector operations from treating invalid scores
  as useful values;
- Reduction validity gating guarantees invalid lanes never affect max or sum;
- normalization gating guarantees invalid output probability is exactly zero.

The design intentionally does not rely only on a finite sentinel to approximate
negative infinity.

This fusion primarily saves movement and control overhead. The first RTL may
still evaluate all eight multipliers before the writeback selects, so its
Vector active cycle count may remain unchanged and its multiplier switching
power may not fall. Later operand gating may reduce invalid-lane switching
power, but it requires measured benefit and must not alter the result.

中文说明：

“Mask融合进Score Scale”具体指同一次Vector写回完成两个动作。每个lane先
得到定点缩放结果，写回前由一个选择器检查该row的mask bit：有效lane写缩放
值，无效lane写`SOFTMAX_NEG_INF`。因此不会再启动一次独立Mask操作，也不会
再次读取和写回Score tile。

当前RTL中`valid_mask=0`时输出为零，不能直接当作Attention Mask，因为零分数
仍会参与Softmax。目标RTL需要让`VSCALE_FIXED`的无效lane写sentinel，同时
Reduction和Normalization继续使用同一row mask精确排除无效lane。该融合主要
节省第二次数据遍历和控制开销；如果八路乘法仍全部翻转，第一版不一定降低
Vector cycle或乘法器动态功耗。

sentinel和row mask不是重复机制：sentinel用于明确保存中间masked-score；
Reduction使用mask保证无效lane不参与max/sum；Normalization使用mask保证
最终概率严格为零。不能只依靠一个有限负数近似数学上的负无穷。

For regular causal and tail cases, an implementation may generate the row mask
from query/key indices instead of storing all row masks, but this is an
encoding optimization. It must preserve the same architectural
`valid_lane_mask[row]` semantics and be compared for area, power, and cycles.

#### Software and hardware ownership

| Layer/module | Required responsibility |
| --- | --- |
| Operator/numerical contract | define visibility semantics, `SOFTMAX_NEG_INF`, all-invalid-row behavior, and output-zero rule |
| Compiler | compose causal/padding/tail rules; skip fully invisible future tiles when multi-tile lowering exists; emit mask metadata |
| Runtime/descriptor | transport mask policy/metadata without interpreting values; reject unsupported masked plans |
| Uop Scheduler | issue row-indexed primitives with an architectural mask reference; no per-lane policy decisions |
| Vector / Scale-Mask | apply fixed scaling and select `SOFTMAX_NEG_INF` for invalid score lanes |
| Reduction | exclude invalid lanes from max/sum and count only valid reduced elements |
| SFU | no mask-policy logic; receives only valid or reviewed sentinel-derived inputs |
| PV / Matrix | consume zero probabilities; future multi-tile planner skips fully invisible tiles |
| PPA | report mask policy, valid elements, skipped tiles, mask movement, and mask-control cycles |

#### Selected interface

The descriptor selects one compact row-mask table. Existing row-indexed
primitives select the matching row implicitly:

```text
descriptor input1 -> local_row_mask_table
row_index         -> valid_lane_mask[row_index]
```

For the current eight-lane tile, each `valid_lane_mask` is eight bits. Eight
rows pack into two 32-bit words. Scale/Mask and Softmax descriptors reference
the same table. The Data mover loads it into reviewed local row-mask registers
before Scheduler launch. RTL must not hard-code causal row patterns.

No descriptor ABI field is added. For the two Attention descriptor op types,
`input1_words=0` means implicit all-valid execution and `input1_words=2` means
`input1_addr` references the packed row-mask table. Command Processor validates
this combination, Data Mover loads and unpacks the two words, and
`row_mask_ready` becomes a Scheduler-launch dependency. Exact hardware
sequencing and error behavior are defined in
`arch/specs/transformer/v1/descriptor_v1.md`.

中文说明：

Mask确实需要通过descriptor告知硬件，但不需要新增descriptor字段。现有
`input1_addr/input1_words`在Scale/Mask和Softmax任务中被定义为row-mask表：
长度为0表示所有lane有效，长度为2表示从地址加载两个mask word。Command
Processor完成校验后，由Data Mover加载到NPU Core本地8个8-bit寄存器；
加载完成前不能启动Uop Scheduler。

No new `MASK` instruction is generated:

```text
Scale/Mask program remains: 8 x VSCALE_FIXED + HALT
Softmax program remains:    112 primitives + HALT
```

This avoids adding serialized Scheduler issue/response overhead. The new PPA
cost is two mask-table words of movement plus lane gating.

#### Correctness corner cases

- Masking happens before row maximum.
- Invalid lanes never contribute to max or sum.
- Invalid probability lanes are exactly zero.
- A row with no valid lane is an error/unsupported input for v1. Runtime and
  Compiler must reject it; Reduction must expose an error if it still reaches
  hardware.
- Padding and tail rules compose with causal visibility using logical AND.
- `SOFTMAX_NEG_INF` is signed int32 minimum and is never treated as a valid
  Reduction/subtraction operand.

#### Expected PPA impact and acceptance

中文说明：

- 新增代价：两个mask word的数据搬运、8-lane门控、Reduction输入筛选以及
  少量控制逻辑；
- 直接收益：避免CPU对Score矩阵进行一次完整读改写，减少SRAM流量和同步；
- 当前限制：`8x8` QK仍会完整计算，因此当前Matrix cycle不会因为mask减少；
- 后续收益：支持多tile后，可以跳过causal Attention中完全不可见的未来
  QK/PV tile，并减少无效Softmax工作。

Expected cost:

- eight valid-lane gates/selects on the current Vector/Reduction path;
- compact row-mask metadata storage/transport;
- command/descriptor fields and verification logic;
- possible critical-path pressure in Reduction input selection.

Expected benefit:

- removes a software-only score-mask read/modify/write pass;
- avoids an extra score-tile synchronization boundary;
- enables future on-chip score residency and fused scheduling;
- permits fully masked future tiles to be skipped in multi-tile causal
  Attention;
- prevents invalid lanes from consuming useful Reduction/SFU work when later
  scheduling supports skipping them.

The feature is accepted only if:

1. causal, padding, and tail goldens pass;
2. invalid lanes do not affect max, sum, probability, or PV output;
3. PPA shows mask-control and mask-movement cost explicitly;
4. masked NPU execution is no slower than the measured CPU-materialization
   fallback for the reviewed workload;
5. synthesized/mapped evidence later confirms that the added gating does not
   invalidate the performance-first timing contract.

#### Remaining completeness work

The implemented single-tile path settled local storage and canonical sentinel
behavior. Remaining work is:

1. precise hardware-visible malformed-descriptor/all-invalid-row reporting;
2. executable tail rows after Scheduler stops issuing invalid physical rows;
3. required PPA comparison against CPU materialization;
4. deciding whether a real workload justifies arbitrary dense masks.

Masking must happen before row max. If an invalid score is simply zeroed after
softmax, it can still affect row max, row sum, and output probability. The
correct behavior is to exclude invalid positions from softmax. The current
unmasked path remains the measured baseline until the accepted architecture
direction is fully specified and implemented.

Current status:

- Compiler emits packed causal row masks and Runtime passes the same table to
  Scale/Mask and Softmax;
- Vector writes `SOFTMAX_NEG_INF` for invalid scaled-score lanes;
- Reduction excludes invalid lanes and normalization writes zero probability;
- causal `S=8,D=8` passes end-to-end RTL execution and golden checking;
- executable tail/all-invalid-row handling remains before generalized decoder
  Attention claims.

### 4. Row Softmax / 逐行Softmax

Formula per row:

```text
row_max[i] = max_j masked_scores[i,j]
delta[i,j] = masked_scores[i,j] - row_max[i]
e[i,j] = exp(delta[i,j])
row_sum[i] = sum_j e[i,j]
P[i,j] = e[i,j] / row_sum[i]
```

Clamp is part of the fixed-point softmax implementation, not a separate
attention semantic. It exists because the SFU EXP target only supports a bounded
input interval:

```text
exp_input[i,j] = clamp(delta[i,j], -256, 0)
```

This clamp happens after row max subtraction. Since `row_max` makes
`delta <= 0`, the high clamp to zero is mainly a safety/saturation rule. The
low clamp to `-256` maps all very small probabilities to the minimum EXP table
entry instead of requiring an unbounded LUT.

Fixed-point v1 target:

```text
exp_input = clamp(masked_scores - row_max, -256, 0)
e_q15 = EXP_Q0_15(exp_input / 32)
sum_q15 = sum(e_q15)
recip_q24 = RECIP_Q24(sum_q15)
P_q15 = requant(e_q15 * recip_q24)
```

Hardware implication:

- reduction engine provides row max and row sum;
- SFU provides EXP and reciprocal;
- vector engine performs subtract, clamp, multiply/requant;
- row length and lane tail behavior must be explicit.

Current status:

- reduction standalone RTL supports max/sum over packed int32 inputs;
- SFU standalone RTL supports coarse EXP and integer reciprocal;
- vector standalone RTL supports subtract, clamp, and simple requant;
- old `npu_v0_compute_cluster` softmax is Phase 0-specific and should not be treated as
  the Transformer attention softmax implementation.

Needed for attention:

- replace coarse SFU EXP with reviewed 257-entry Q0.15 LUT or keep reports
  clearly labeled as bring-up accuracy;
- define reciprocal Q format and normalization shift;
- add row-softmax golden vectors and tolerance;
- add valid/ready/counter semantics before scheduler integration;
- report reduction/SFU/vector cycles separately.

中文说明：

Softmax是Compiler生成的逐行micro-kernel，由Reduction求最大值和求和、
Vector执行减法/截断/归一化、SFU执行EXP和倒数。Uop Scheduler负责取指、
译码、发射和等待完成，各计算模块不负责决定下一条Softmax原语。

### 5. Probability-Value Matmul / 概率与Value矩阵乘

Formula:

```text
O[i,d] = sum(j=0..S-1) P[i,j] * V[j,d]
```

`Probability-Value` means the final attention weighted sum: the softmax
probability matrix `P` weights the value matrix `V`. This is often abbreviated
as `P*V` or attention-value matmul. `P` is not a learned weight matrix; it is
the runtime probability distribution produced from QK scores.

The mixed-precision issue comes directly from the datatypes:

```text
P[i,j]  = softmax probability, naturally fractional in [0, 1]
V[j,d]  = int8 value activation/cache element
O[i,d]  = sum of fractional probability times int8 value
```

The current matrix engine accepts int8 x int8. But the numerically natural
fixed-point probability is Q0.15, not int8. Therefore `P*V` has two possible
hardware-facing representations.

Initial implementation options:

1. Convert or requant `P_q15` to int8 and reuse the int8 matrix engine:

   ```text
   P_i8 = requant(P_q15)
   O_i32 = P_i8 * V_i8
   ```

   This is current-matrix compatible but loses probability precision and needs
   a reviewed scale/zero-point policy.

2. Add mixed precision support for Q0.15 probability times int8 value:

   ```text
   O_acc[i,d] = sum_j P_q15[i,j] * V_i8[j,d]
   O[i,d] = O_acc[i,d] >>> 15
   ```

   This better matches attention math but requires a new weighted-sum or
   mixed-precision matrix path.

3. Model PV until a reviewed mixed-precision or requant path exists.

The first option is simplest for current RTL compatibility but loses
probability precision. The second option better represents attention but adds
hardware cost and PPA implications.

Current status:

- matrix engine only supports int8 x int8 -> int32;
- vector engine can shift/clamp values but does not define probability-to-int8
  attention policy;
- no mixed Q0.15 x int8 matrix datapath exists.

Needed for attention:

- choose the first executable PV policy before RTL work:
  `P_q15_to_i8_then_matmul` or `mixed_q15_i8_weighted_sum`;
- if using int8 probability, define rounding/clamp and expected error;
- if using mixed precision, update matrix/vector docs, area proxy, and tests.

中文说明：

PV阶段计算`P × V`得到Attention输出。P是运行时Softmax概率，不是模型
权重。当前路径使用Q0.15概率乘int8 Value并输出int32；较大的序列维度同样
需要沿K维分块累加，较大的输出维度需要多个输出tile。

## Current Implementation Inventory / 当前实现清单

| Capability | Current implementation | Attention readiness |
| --- | --- | --- |
| int8 tile matmul | `hw/npu_core/rtl/matrix/matmul_array.sv` | ready for QK `S=8,D=8`; PV only if probability is int8 |
| K-stream accumulation | wrapper/core K-stream descriptor path | useful for larger `D`; not attention-specific |
| accumulator file | `matrix/accumulator_file.sv` | stores matrix tile accumulations; score/probability lifetime needs memory contract |
| vector primitive ops | `vector/vector_engine.sv` | standalone only; needs mask/requant details and scheduler path |
| reduction primitive ops | `reduction/reduction_engine.sv` | standalone only; row softmax length/mask/latency needs contract |
| SFU primitive ops | `sfu/sfu_lut.sv` | coarse bring-up only; EXP/RECIP detail must be upgraded |
| primitive integration wrapper | `transformer_primitive_engines.sv` | standalone test integration only |
| core command processor/runtime descriptors | `npu_v0_core_system.sv` | matmul/k-stream focused; no command-list attention sequence |
| perf/PPA | perf CSR snapshots and L0 report | matmul/data mover visible; attention group counters missing |

该表用于区分已经可以作为RTL证据的能力和仍需设计或验证的能力。任何尚未
实现的能力都不能仅依靠模型估算标记为已执行。

## Planned Execution Stages / 计划执行阶段

The numerical iteration plan is defined in
`docs/design/transformer/attention_numerical_v1.md`. The execution stages below
are intentionally compatible with starting from simplified primitives and later
upgrading score scale, EXP, RECIP, normalization, mask, and PV policy. The key
rule is that each simplification must be explicit in workload metadata and
fixture contract IDs.

### Stage A: Model And Documentation

- Add attention golden functions.
- Add attention workload manifest entries.
- Keep softmax/PV model-only where RTL cannot yet execute them.
- Define PPA grouping and expected fields.
- Add numerical policy fields for any simplified behavior, such as
  `score_scale=shift_approx` or `sfu_exp=bringup_9_segment`.

Exit criteria:

- Python tests validate QK, row softmax, PV policy, and full attention output.
- PPA report can include attention model-only sections without pretending they
  are measured.

### Stage B: Measured QK Matmul

- Execute `attention_qk_s8_d8` using current K-stream or single-tile matmul.
- Track Q/K bytes, output score bytes, effective MACs, and matrix utilization.

Exit criteria:

- `perf-report` and `ppa-l0-report` show measured QK cycles.
- Existing CNN and operator smoke regressions still pass.

### Stage C: Scheduler-Visible Softmax Primitives

- Connect vector/reduction/SFU primitive issue path.
- Add row-softmax micro-kernel tests.
- Report vector/reduction/SFU active cycles.
- Start with simplified primitives only if reports label them as bring-up.
- Upgrade to target attention softmax before using output accuracy or SFU PPA as
  architecture evidence.

Exit criteria:

- row softmax has RTL-measured primitive cycles and matches the active
  numerical contract stage-by-stage.

### Stage D: PV Path

- Choose and implement the reviewed PV policy.
- Report PV MACs/cycles separately.
- Integrate full `attention_prefill_s8_d8` as measured sequence.

Exit criteria:

- QK, softmax, and PV are all represented in one attention workload group.

### Stage E: Decode And KV Traffic

- Add decode shape such as `S=1, context=32, D=16`.
- Keep KV cache as external-memory accounting until RTL streamer is justified.
- Use PPA evidence to decide whether true GEMV/skinny-GEMM or KV streaming is
  the next hardware priority.

## Non-Goals / 非目标

- No dedicated attention RTL macro in v1.
- No fused attention pipeline.
- No full decoder block until attention sequence evidence is stable.
- No multi-head hardware fusion.
- No real LPDDR/KV cache controller.

中文说明：

v1阶段不会增加独立Attention宏模块、完整融合流水线、多头硬件融合或真实
LPDDR/KV控制器。先保证共享原语路径正确、可测量，再根据PPA证据决定是否
引入融合硬件。
