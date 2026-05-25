# Workloads

Workloads are first-class architecture evaluation inputs rather than only test
fixtures.

```text
manifests/     versioned shape, precision, scenario, and required-metric definitions
smoke/         minimal operator checks
cnn/           compatibility and existing system-regression workloads
transformer/   long-term LLM inference architecture drivers
```

Existing MNIST implementation and fixtures remain in their current tested
locations while manifests and new Transformer work enter here. Migration of
working fixture code should happen only when it reduces duplication or enables
a required comparison.

See `docs/design/transformer_workloads.md`.
