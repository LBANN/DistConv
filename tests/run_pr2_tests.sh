#!/bin/bash
#
# PR2 Test Runner
# Runs tests for DistConvDDP bucket communication hook
#
# Usage:
#   ./run_pr2_tests.sh [test_selector]
#
# Examples:
#   ./run_pr2_tests.sh                                    # Run all PR2 tests
#   ./run_pr2_tests.sh TestBucketHookCorrectness         # Run specific test class
#   ./run_pr2_tests.sh test_spatial_topology_2ranks      # Run specific test

set -e  # Exit on error

export PYTORCH_MIOPEN_SUGGEST_NHWC=1

# Determine number of processes based on test
if [[ "$1" == *"4ranks"* ]] || [[ "$1" == *"mixed"* ]]; then
    NPROC=4
else
    NPROC=2
fi

echo "=================================================="
echo "PR2 DDP Bucket Hook Tests"
echo "=================================================="
echo "Processes: $NPROC"
echo "Test selector: ${1:-all tests}"
echo "=================================================="
echo ""

# Run with torch.distributed.run (torchrun alternative)
if [ -z "$1" ]; then
    # Run all PR2 tests
    python -m torch.distributed.run \
        --standalone \
        --nproc-per-node=$NPROC \
        -m pytest tests/test_ddp_bucket_hook.py -v
else
    # Run specific test
    python -m torch.distributed.run \
        --standalone \
        --nproc-per-node=$NPROC \
        -m pytest tests/test_ddp_bucket_hook.py::$1 -v
fi

echo ""
echo "=================================================="
echo "PR2 tests completed"
echo "=================================================="
