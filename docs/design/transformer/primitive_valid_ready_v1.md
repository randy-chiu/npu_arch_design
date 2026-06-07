# Primitive Valid/Ready v1 Design

## Scope

This document defines the accepted issue/accept/complete contract for
Transformer primitive engines: vector, reduction, and SFU. It is a design
contract implemented first through compatibility shims around the current
start/done bring-up engines.

## What "primitive" means in this project

`primitive` means a small hardware operation that can be issued directly to
one compute engine. It is the lowest scheduler-visible unit of useful work in
this design. The word does not mean that the RTL is trivial; it means that the
operation is a building block rather than a complete model operator.

Examples:

| Level | Example | Meaning |
| --- | --- | --- |
| Primitive operation | vector subtract, reduce max, SFU EXP, matrix tile multiply | one directly issued engine operation |
| Micro-kernel | one softmax row | ordered sequence of reduce max, vector subtract, EXP, reduce sum, RECIP, and vector scale primitives |
| Attention stage | QK, scale/mask, softmax, or PV | one compiler/runtime-visible stage, potentially containing many primitive operations |
| Composite operator | scaled dot-product attention | QK -> scale/mask -> softmax -> PV |

For example, `softmax(scores)` is not implemented as one dedicated softmax
hardware macro. The scheduler builds it from shared primitives:

```text
REDUCE_MAX -> VEC_SUB -> VEC_CLAMP -> SFU_EXP x lanes
           -> REDUCE_SUM -> SFU_RECIP -> VEC_SCALE
```

Therefore this valid/ready work is called a primitive interface design: it
defines how the scheduler safely issues each small operation to the vector,
reduction, or SFU engine and receives its result.

## Why start/done is insufficient

The current `start/done` mechanism is useful for isolated bring-up tests where
the testbench knows the exact engine latency and always consumes the result
immediately. It does not define enough behavior for a scheduler coordinating
multiple engines with different or variable latencies.

### Problem 1: a start pulse can be lost while the engine is busy

With only `start`, the producer has no explicit signal saying whether the
engine can accept a command:

```text
cycle       0       1       2       3
engine      busy    busy    busy    done
start       0       1       0       0
```

If the scheduler pulses `start` in cycle 1, the interface does not say whether
the command was accepted, ignored, or corrupted the running command. Avoiding
this requires the scheduler to know each engine's internal busy state and exact
latency, coupling scheduler logic to implementation details.

This occurs in attention softmax when several lane EXP operations must be sent
to an SFU. A future RECIP or pipelined EXP implementation may take a different
number of cycles from the current bring-up RTL. A fixed pulse schedule would
then lose or overlap requests.

With valid/ready, the producer holds the command until:

```text
cmd_fire = cmd_valid && cmd_ready
```

There is one unambiguous acceptance event. The scheduler does not need to guess
the engine's latency or busy state.

### Problem 2: a one-cycle done pulse can be lost

With only `done`, the engine assumes the consumer can always capture the result
on the done cycle:

```text
cycle       0       1       2       3
done        0       0       1       0
consumer    ready   busy    busy    ready
```

If the scheduler, output buffer, or next engine is busy in cycle 2, the
one-cycle completion event and result can be lost. The producer cannot wait
because there is no response back-pressure signal.

This matters when a reduction result must feed SFU RECIP, or an SFU result must
wait for a vector normalization command. Those consumers may be occupied by a
previous operation.

With valid/ready, the engine asserts `rsp_valid` and holds both `rsp_valid` and
the result stable until:

```text
rsp_fire = rsp_valid && rsp_ready
```

The result therefore remains available across any number of consumer stall
cycles.

### Problem 3: start/done does not provide a stable counter contract

For performance and PPA evidence, the design needs to distinguish:

- useful engine work;
- producer waiting because the engine cannot accept input;
- engine/result waiting because the consumer cannot accept output;
- true idle time.

A `start` pulse and a `done` pulse only identify two isolated events. They do
not identify the cycles between them or explain why progress stopped. Counting
internal FSM states would also make reports dependent on each engine's private
implementation.

Valid/ready exposes architectural events with consistent meanings:

```text
input stall:   cmd_valid && !cmd_ready
output stall:  rsp_valid && !rsp_ready
accepted op:   cmd_valid && cmd_ready
retired op:    rsp_valid && rsp_ready
```

These definitions remain valid if an engine later changes from one cycle to
multiple cycles.

### Problem 4: fixed-latency sequencing does not compose

One attention softmax row combines reduction, vector, and SFU engines. Even if
each standalone engine passes a `start/done` test, composing them requires
answers to the following questions:

- Can the next primitive be issued before the previous result is consumed?
- What happens when vector and SFU complete on the same cycle?
- Where is a result stored if the next engine is busy?
- Can reset discard an accepted operation or replay a stale result?

`start/done` does not answer these questions. Valid/ready plus an explicit
response slot establishes ownership: a command belongs to the engine after
`cmd_fire`, and a result belongs to the consumer only after `rsp_fire`.

## Why the selected mechanism works

The first implementation uses one command in flight and one held response
slot per compatibility shim:

```text
scheduler command
  -> cmd_valid/cmd_ready acceptance
  -> shim locks command payload
  -> existing start/done engine executes
  -> shim stores result and asserts rsp_valid
  -> result stays stable until rsp_ready
```

This mechanism is intentionally conservative:

- no command can be lost because acceptance requires `cmd_fire`;
- command fields cannot change after acceptance because the shim locks them;
- no result can be lost because the response slot holds it until `rsp_fire`;
- back-pressure propagates naturally because a full response slot deasserts
  `cmd_ready`;
- counters derive from visible handshake events rather than engine-specific
  internal states;
- existing start/done RTL and SoC regressions remain unchanged behind the shim.

The single-entry shim does not improve throughput. Its purpose is to establish
a correct scheduler contract first. Later implementations may pipeline engines
or allow multiple commands in flight while preserving the same external
acceptance and response rules.

## Current status

Current primitive RTL is standalone bring-up RTL:

- `start` launches one operation.
- `done` pulses when the result is available.
- `active` mirrors the current operation.
- There is no `valid/ready` back-pressure.
- There are no real stall counters.

`hw/npu_core/rtl/primitive_handshake_shims.sv` remains the standalone directed
engine-contract test vehicle. The production Scheduler-to-Compute-cluster path
now uses the same one-command/one-response valid-ready ownership contract.
Compute cluster may adapt that architectural channel to current start/done
engines internally, but start/done is no longer the Scheduler contract.

## Target interface

Each primitive engine should expose an issue channel and a completion channel.

Issue side:

```text
cmd_valid
cmd_ready
cmd_* payload fields
```

Completion side:

```text
rsp_valid
rsp_ready
rsp_* result fields
```

Minimal integration may tie `rsp_ready = 1` until a downstream queue exists.
The contract should still define `rsp_ready` so later scheduler integration
does not need another interface rewrite.

## Handshake semantics

An operation is accepted on a rising clock edge when:

```text
cmd_fire = cmd_valid && cmd_ready
```

Payload fields must remain stable while `cmd_valid = 1` and `cmd_ready = 0`.

Result is transferred when:

```text
rsp_fire = rsp_valid && rsp_ready
```

The engine must hold `rsp_valid` and result payload stable while
`rsp_valid = 1` and `rsp_ready = 0`.

## Ordering

Initial v1 engines are in-order:

- One accepted command produces one response.
- Responses retire in command acceptance order.
- No command ID is required for standalone primitive engines.

If a later engine allows multiple in-flight commands, that engine must either
preserve in-order response or add an explicit response tag before integration.

## Engine readiness

`cmd_ready` means the engine can accept a command on the current cycle.

For a single-entry, non-pipelined implementation:

```text
cmd_ready = !busy || accepting_pipeline_overlap
```

For a single-cycle combinational/vector bring-up equivalent:

```text
cmd_ready = 1 when response path can accept the result
```

If `rsp_valid = 1` and `rsp_ready = 0`, a simple engine should deassert
`cmd_ready` unless it has separate buffering for another response.

## Latency model

Each engine spec must declare:

- acceptance latency: command accepted on `cmd_fire`;
- response latency: cycles from `cmd_fire` to `rsp_valid`;
- whether a new command can be accepted every cycle.

Initial target assumptions:

| Engine | Response latency | Initiation interval |
| --- | --- | --- |
| Vector v1 | 1 cycle | 1 cycle if response is ready |
| Reduction v1 bring-up | 1 cycle | 1 cycle if response is ready |
| SFU EXP 257-entry LUT | 1 cycle unless ROM registration requires 2 | 1 cycle if response is ready |
| SFU RECIP/RSQRT bring-up | TBD before production implementation | TBD |

Any implementation with different latency must update the relevant engine spec
and tests first.

The compatibility shims have a two-cycle `cmd_fire` to `rsp_valid` latency
because they register the command, pulse the underlying start/done engine, and
then capture its result. They do not overlap commands. This is compatibility
behavior, not the target initiation interval for native valid/ready engines.

## active/stall/idle counters

Counters should be defined from handshake events, not inferred from internal
implementation details.

Definitions:

| Counter | Increment condition |
| --- | --- |
| `*_active_cycles` | engine has accepted work not yet retired, or accepts and retires same-cycle work |
| `*_input_stall_cycles` | `cmd_valid = 1` and `cmd_ready = 0` |
| `*_output_stall_cycles` | `rsp_valid = 1` and `rsp_ready = 0` |
| `*_idle_cycles` | no active work, no accepted command, no valid response |

For the first CSR exposure, `stall_cycles_by_engine` may be the sum of input
and output stall cycles. Reports must label whether stall is measured or still
modeled.

## Reset behavior

On reset:

- `cmd_ready` should return to a known state.
- `rsp_valid` must clear.
- No stale result may be emitted after reset.
- Active/stall counters reset through the existing CSR/reset policy, not
  implicitly by this primitive interface unless the wrapper defines that reset.

## Compatibility with start/done bring-up

A compatibility shim may map current tests onto the new interface:

```text
start -> cmd_valid pulse
cmd_ready must be high for acceptance
done <- rsp_valid pulse when rsp_ready is tied high
```

The old start/done interface remains only inside current engine adapters and
standalone engine RTL. It is not the production Scheduler contract.

## Verification plan

Required directed tests:

- Accept one command when `cmd_valid && cmd_ready`.
- Hold command payload stable across one or more input stall cycles.
- Hold response payload stable when `rsp_ready = 0`.
- Count input stall cycles.
- Count output stall cycles.
- Confirm no response appears after reset without a new command.
- Confirm existing primitive operation results match the current RTL model.

These tests are implemented in
`hw/npu_core/tb/primitive_handshake_tb.sv` and run under
`make primitive-engines-sim`.

## Known risks

- A single valid/ready contract must cover both single-cycle vector ops and
  longer SFU/RECIP/RSQRT paths without hiding latency.
- Counter increments must be specified before CSR/report plumbing, otherwise
  perf reports may mix measured and modeled semantics.
- Scheduler integration should not assume all primitive engines have identical
  latency.
