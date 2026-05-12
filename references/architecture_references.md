# Architecture References

This directory tracks architecture references and design ideas. The project
should use these references as input for later architecture iterations, not as
requirements for the minimal Phase 0/Phase 1 system.

## Update Policy

- Add new papers, whitepapers, technical blogs, and vendor docs as short
  annotated entries.
- Keep links to the original sources instead of copying paper text.
- Extract reusable ideas into `candidate_ideas.md`.
- Promote an idea to implementation only through a hardware spec change and a
  passing verification run.
- Manual refresh cadence: review references before each major architecture
  iteration. Fully automatic background updates are not available in this
  environment, so refreshes should be triggered during planning turns.

## Initial Reference Set

### Google TPU v1

- Source: "In-Datacenter Performance Analysis of a Tensor Processing Unit",
  Jouppi et al., ISCA 2017.
- Link: https://doi.org/10.1145/3079856.3080246
- Accessible copy found during research:
  https://users.cs.duke.edu/~lkw34/papers/tpu-isca2017.pdf
- Topic: systolic matrix unit, inference accelerator, software-managed memory.
- Useful ideas:
  - Start with a narrow domain-specific ISA.
  - Use a large matrix unit with explicit data movement.
  - Treat memory bandwidth and utilization as first-class design metrics.
- Project status: `candidate`.

### Google TPU v4

- Source: "TPU v4: An Optically Reconfigurable Supercomputer for Machine
  Learning with Hardware Support for Embeddings".
- Link: https://arxiv.org/abs/2304.01433
- Topic: large-scale accelerator interconnect, embeddings, pod-level topology.
- Useful ideas:
  - Separate single-chip architecture decisions from system-scale topology.
  - Interconnect flexibility matters once scaling beyond one device.
- Project status: `watch`.

### NVIDIA Ampere

- Source: NVIDIA Ampere Architecture overview.
- Link: https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/
- Topic: tensor cores, TF32/BF16/INT8/INT4, structured sparsity, MIG.
- Useful ideas:
  - Mixed precision should be explicit in ISA and compiler metadata.
  - Sparse acceleration must be coupled to compiler-visible constraints.
- Project status: `candidate`.

### NVIDIA Hopper

- Source: NVIDIA Hopper Architecture page and technical blog.
- Links:
  - https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/
  - https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Topic: transformer engine, FP8, async data movement, high-bandwidth scale-out.
- Useful ideas:
  - Add low-precision formats only after the base numerical path is verified.
  - Async copies and double buffering should be measured with utilization
    counters before being added to hardware complexity.
- Project status: `candidate`.

### NVIDIA Blackwell

- Source: NVIDIA Blackwell Architecture page.
- Link: https://www.nvidia.com/en-gb/data-center/technologies/blackwell-architecture/
- Topic: second-generation transformer engine, FP4/microscaling, chip-to-chip
  interconnect.
- Useful ideas:
  - Future quantization design should include per-tensor or micro-tensor scale
    metadata instead of assuming one global scale.
- Project status: `watch`.

### AMD CDNA / MI300

- Sources:
  - AMD CDNA Architecture page: https://www.amd.com/en/technologies/cdna.html
  - AMD MI300 page: https://www.amd.com/en/products/accelerators/instinct/mi300.html
- Topic: matrix cores, HBM bandwidth, chiplets, unified CPU/GPU package.
- Useful ideas:
  - Memory capacity and bandwidth need to be reported alongside peak compute.
  - Chiplet and package-level ideas belong to a later multi-tile phase.
- Project status: `candidate`.

### Eyeriss

- Source: "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep
  Convolutional Neural Networks".
- Link: https://www.rle.mit.edu/eyeriss-an-energy-efficient-reconfigurable-accelerator-for-deep-convolutional-neural-networks/
- DOI: https://doi.org/10.1109/JSSC.2016.2616357
- Topic: row-stationary dataflow, data reuse, data movement energy.
- Useful ideas:
  - Data movement should be tracked from the first PPA model.
  - Compiler tiling should optimize locality, not only MAC occupancy.
- Project status: `candidate`.

### Graphcore IPU

- Source: Graphcore IPU Programmer's Guide.
- Link: https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html
- Topic: many-tile architecture, local SRAM, explicit exchange, BSP execution.
- Useful ideas:
  - Explicit compute/sync/exchange phases can simplify deterministic scheduling.
  - Distributed local memory is powerful but too complex for Phase 0.
- Project status: `watch`.

### Tenstorrent Wormhole

- Source: Tenstorrent Wormhole hardware page.
- Link: https://open.tenstorrent.com/hardware/wormhole
- Topic: spatial cores, local cache, NoC, RISC-V control, open software stack.
- Useful ideas:
  - Keep low-level control programmable without hiding data movement.
  - NoC ideas should wait until shared bus limits are measured.
- Project status: `watch`.

