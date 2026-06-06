# NPU Subsystem

This directory owns the primary PPA top for the NPU implementation.

```text
npu_subsystem_top
  = Host wrapper
  + NPU core system
      + command processor
      + data mover
      + compute cluster
  + exposed external memory boundary
```

The simulation SoC remains under `hw/soc/`; it is not the primary NPU area or
power boundary because it includes CPU and bring-up memory/peripheral models.

Design contract:

```text
docs/design/npu_subsystem.md
docs/design/ppa_methodology.md
```
