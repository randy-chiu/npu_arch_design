# NPU Core Programs

Checked-in NPU-core program descriptions can live here when they are part of
the design source rather than generated test artifacts.

The target direction is that fixed operator programs, such as the micro-op
stream for a matmul operator shape, are produced by NPU compiler/assembler
tools and consumed by the NPU runtime. In the current Phase 0 smoke test these
program words are generated into firmware data and copied by the CPU into the
NPU wrapper program window. Later they should be emitted as NPU program streams
that firmware places in SRAM for the NPU wrapper/core to fetch by address.
