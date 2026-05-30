# Primitive Valid/Ready v1 Design

## Scope

This document defines the proposed issue/accept/complete contract for
Transformer primitive engines: vector, reduction, and SFU. It is a design
contract only. Current RTL still uses the start/done bring-up interface.

No RTL implementation should start from this document until review is complete.

## Current status

Current primitive RTL is standalone bring-up RTL:

- `start` launches one operation.
- `done` pulses when the result is available.
- `active` mirrors the current operation.
- There is no `valid/ready` back-pressure.
- There are no real stall counters.

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

The old start/done interface should not remain the production scheduler
contract once valid/ready is implemented.

## Verification plan

Required directed tests:

- Accept one command when `cmd_valid && cmd_ready`.
- Hold command payload stable across one or more input stall cycles.
- Hold response payload stable when `rsp_ready = 0`.
- Count input stall cycles.
- Count output stall cycles.
- Confirm no response appears after reset without a new command.
- Confirm existing primitive operation results match the current RTL model.

## Known risks

- A single valid/ready contract must cover both single-cycle vector ops and
  longer SFU/RECIP/RSQRT paths without hiding latency.
- Counter increments must be specified before CSR/report plumbing, otherwise
  perf reports may mix measured and modeled semantics.
- Scheduler integration should not assume all primitive engines have identical
  latency.
