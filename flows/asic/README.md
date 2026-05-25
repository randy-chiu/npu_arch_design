# ASIC Flows

This directory is reserved for repeatable ASIC-oriented implementation and
power-estimation flows.

Initial intent:

```text
openroad/
  common/              shared scripts and report extraction
  targets/sky130hd/    first public ASIC comparison target
  tops/npu_core/       core-only breakdown flow
  tops/npu_subsystem/  primary PPA flow
  tops/soc_reference/  optional system-reference flow
```

The first implementation target is `sky130hd` as an open, repeatable estimate.
Generated flow output belongs under `build/ppa/`, not under this directory.

The checked-in OpenROAD Flow Scripts (ORFS) design inputs follow the official
design configuration convention: each top provides `config.mk` and
`constraint.sdc`, and the config identifies `DESIGN_NAME`, `PLATFORM`,
`VERILOG_FILES`, `VERILOG_INCLUDE_DIRS`, and `SDC_FILE`.

Before invoking ORFS, generate the current spec includes and check elaboration:

```text
make npu-subsystem-elab
```

Official references:

- https://openroad-flow-scripts.readthedocs.io/en/stable/tutorials/FlowTutorial.html
- https://openroad-flow-scripts.readthedocs.io/en/stable/user/FlowVariables.html
