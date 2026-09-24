# PR2 Execution Guide

## Implementation Complete

The DistConvDDP bucket communication hook refactor is implemented and ready for testing.

**Branch:** `compile/ddp-hook`  
**Base commit:** `084b20a` (main)

## Changes Made

### 1. Production Code: `distconv/ddp.py`

**Before:** 50 lines with per-parameter gradient hooks  
**After:** 149 lines with bucket communication hook

**Key changes:**
- Added `_DistConvCommHookState` dataclass to hold hook state
- Implemented `_distconv_comm_hook()` with async all-reduce pattern
- Refactored `__init__()` to register bucket hook instead of per-parameter hooks
- Added validation for gradient divisor and topology compatibility

### 2. Test Suite: `tests/test_ddp_bucket_hook.py`

**New file:** 588 lines

**Test coverage:**
- `TestBucketHookCorrectness` - gradient correctness across topologies
- `TestAccumulation` - gradient accumulation with `no_sync()`
- `TestCompilation` - `torch.compile()` compatibility
- `TestCompatibility` - API compatibility and validation

### 3. Demo: `examples/pr2_ddp_demo.py`

**New file:** 367 lines

**Features:**
- Supports spatial, data-parallel, and mixed topologies
- Tests eager and compiled modes
- Demonstrates gradient accumulation
- Exports JSON results

### 4. Test Runner: `tests/run_pr2_tests.sh`

**New file:** Standalone PR2 test runner

## Execution Commands

### Prerequisites

**Must run on compute node with GPU allocation** (not login node):

```bash
# Example SLURM allocation (adjust for your system)
srun -N 1 -n 4 --gpus-per-node=4 --pty bash

# Or use existing allocation
# Verify GPUs available: rocm-smi or nvidia-smi
```

### Test Execution

#### 1. Core Correctness Tests (2-rank)

```bash
cd /usr/workspace/dudouit1/DistConv

# Test spatial topology (W=2, S=2, D=1)
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py::TestBucketHookCorrectness::test_spatial_topology_2ranks -v

# Test data-parallel topology (W=2, S=1, D=2)
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py::TestBucketHookCorrectness::test_data_parallel_topology_2ranks -v

# Test bucket view modes
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py::TestBucketHookCorrectness::test_bucket_view_modes -v
```

#### 2. Accumulation Tests (2-rank)

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py::TestAccumulation -v
```

#### 3. Compilation Tests (2-rank)

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py::TestCompilation -v
```

#### 4. Compatibility Tests (2-rank)

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py::TestCompatibility -v
```

#### 5. All PR2 Tests

```bash
# Run all tests (2-rank)
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest tests/test_ddp_bucket_hook.py -v

# Or use the test runner script
cd tests && ./run_pr2_tests.sh
```

### Demo Execution

#### 1. Spatial Topology (2-rank: W=2, S=2, D=1)

```bash
cd /usr/workspace/dudouit1/DistConv/examples

python -m torch.distributed.run --standalone --nproc-per-node=2 \
    pr2_ddp_demo.py --topology spatial

# Check results
cat pr2_demo_results_spatial_w2.json
```

#### 2. Data-Parallel Topology (2-rank: W=2, S=1, D=2)

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    pr2_ddp_demo.py --topology data

# Check results
cat pr2_demo_results_data_w2.json
```

#### 3. Mixed Topology (4-rank: W=4, S=2, D=2)

**Requires 4 GPUs:**

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
    pr2_ddp_demo.py --topology mixed

# Check results
cat pr2_demo_results_mixed_w4.json
```

### Backward Compatibility Verification

```bash
# Verify existing tests still pass with new implementation
cd /usr/workspace/dudouit1/DistConv/tests

python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest test_ddp_with_distconv.py -v
```

## Environment Setup

### Required Environment Variables

```bash
# For ROCm systems (MIOpen caching)
export PYTORCH_MIOPEN_SUGGEST_NHWC=1

# Optional: Set cache directories if needed
export TRITON_CACHE_DIR=/path/to/writable/cache/triton
export PYTORCH_INDUCTOR_CACHE_DIR=/path/to/writable/cache/inductor
```

### Python Environment

Current environment:
- Python: 3.13.2
- PyTorch: 2.7.0+rocm6.3
- Target compatibility: PyTorch >= 2.5

## Expected Results

### Core Correctness

**Spatial topology (W=2, S=2, D=1):**
- Gradients should be summed across spatial shards
- No averaging (divisor D=1)

**Data-parallel topology (W=2, S=1, D=2):**
- Gradients should be averaged across data replicas
- Divisor D=2

**Mixed topology (W=4, S=2, D=2):**
- Spatial shards summed, data replicas averaged
- Divisor D=2

### Accumulation

- `no_sync()` should prevent communication
- Synchronized backward should reduce accumulated gradients correctly
- Multiple accumulation windows should produce consistent results

### Compilation

- `torch.compile(backend="inductor")` should succeed
- Compiled gradients should match eager mode
- No recompilation on repeated steps with same shapes

### Compatibility

- Explicit `grad_reduction_factor` should work
- `ParallelStrategy`-derived factor should work
- Invalid divisors should be rejected with clear error messages
- Topology mismatches should be rejected

## Known Limitations

### Out of Scope for PR2

- DCTensor tracing and compiled halo exchange
- Compiled autograd
- Subgroup support beyond default world group
- DDP-native mixed precision (uses conflicting hook)

### Unsupported Patterns

- Repeated synchronized backward without clearing gradients
- DDP-native mixed precision (GradientCompressionReducer conflicts)
- Optimizer-in-backward hooks

### Supported Patterns

- Standard accumulation: `zero_grad()` → `no_sync()` × N → synchronized backward → `step()`
- Ordinary autocast with BF16/FP16 and GradScaler
- Frozen parameters, `find_unused_parameters`, `static_graph`

## Troubleshooting

### Cannot run on login node

If you see "no torchrun" or GPU access errors:
- You are on a login node
- Request compute node allocation first
- Example: `srun -N 1 -n 4 --gpus-per-node=4 --pty bash`

### NCCL timeout errors

If tests hang or timeout:
- Check all ranks have access to GPUs
- Verify `CUDA_VISIBLE_DEVICES` or equivalent is set correctly
- Check firewall/network connectivity between ranks

### MIOpen cache errors

If you see MIOpen database errors:
- Set `PYTORCH_MIOPEN_SUGGEST_NHWC=1`
- Ensure cache directory is writable
- May need to clear cache: `rm -rf ~/.cache/miopen`

### Import errors

If modules cannot be imported:
- Ensure you're in the repository root or examples/ directory
- Check Python path includes DistConv package
- Verify: `python -c "from distconv import DistConvDDP; print('OK')"`

## Next Steps After Testing

1. **Collect results:**
   - Save pytest output to file
   - Collect JSON results from demo runs
   - Note any failures or issues

2. **Document results:**
   - Update `docs/PR2_results.md` with findings
   - Include environment details
   - Report status per test category

3. **Create PR:**
   - Commit changes to `compile/ddp-hook` branch
   - Push to remote
   - Create pull request with results

## Contact

For questions about execution or results, consult the PR2 implementation plan at:
`.claude/plans/implement-pr-2-snoopy-pine.md`
