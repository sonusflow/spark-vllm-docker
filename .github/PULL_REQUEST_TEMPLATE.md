## Summary

<!-- Brief description of changes -->

## Type of Change

- [ ] New recipe
- [ ] New mod/patch
- [ ] Bug fix
- [ ] Script/tooling improvement
- [ ] Documentation

## Automated Checks

CI will run automatically:
- [ ] Static validation (ShellCheck, recipe schema, secrets scan)
- [ ] Recipe dry-run tests
- [ ] GPU smoke test (on merge to main)

## Manual Validation Checklist

For recipe or mod changes, verify before merging:

### Output Quality
- [ ] Model produces coherent, non-garbage output
- [ ] Tool calling works (if `--enable-auto-tool-choice` is used)
- [ ] Reasoning output parses correctly (if `--reasoning-parser` is used)

### Cluster (if multi-node recipe)
- [ ] All nodes join Ray cluster successfully
- [ ] Model shards load on every node
- [ ] No NCCL timeout errors in logs

### Performance
- [ ] Benchmark results (paste below or link):
  - Single user tok/s: ___
  - Concurrent (4) aggregate tok/s: ___
- [ ] No significant regression from previous benchmarks

### Environment
- [ ] NVIDIA driver version tested: ___
- [ ] Number of nodes: ___
- [ ] Container image used: ___
