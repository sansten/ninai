#!/bin/bash

# Performance Testing Suite Runner
# 
# This script runs all performance tests and generates a comprehensive report.
# It can be used for:
# - Local development baseline testing
# - Pre-deployment validation
# - Capacity planning
# - Regression detection
#
# Usage:
#   ./run-performance-tests.sh [environment] [test-type]
#
# Examples:
#   ./run-performance-tests.sh                    # Run all tests against localhost
#   ./run-performance-tests.sh staging             # Run all tests against staging
#   ./run-performance-tests.sh prod queue          # Run only queue tests against prod
#   ./run-performance-tests.sh local all           # Run all tests, save results

set -e

# Configuration
ENVIRONMENT=${1:-local}
TEST_TYPE=${2:-all}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Environment mappings
case $ENVIRONMENT in
  local)
    BASE_URL="http://localhost:8000"
    echo "📊 Running tests against LOCAL (http://localhost:8000)"
    ;;
  staging)
    BASE_URL="https://api-staging.ninai.com"
    echo "📊 Running tests against STAGING"
    ;;
  prod)
    BASE_URL="https://api.ninai.com"
    echo "⚠️  WARNING: Running tests against PRODUCTION"
    echo "⚠️  Make sure you understand the impact of load tests!"
    read -p "Press ENTER to continue or Ctrl+C to cancel: "
    ;;
  *)
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Valid options: local, staging, prod"
    exit 1
    ;;
esac

# Get auth token
if [ -z "$AUTH_TOKEN" ]; then
  echo "❌ AUTH_TOKEN environment variable not set"
  echo "Set it with: export AUTH_TOKEN=your-token"
  exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  NINAI PERFORMANCE TESTING SUITE"
echo "═══════════════════════════════════════════════════════════════"
echo "Environment:  $ENVIRONMENT"
echo "Base URL:     $BASE_URL"
echo "Test Type:    $TEST_TYPE"
echo "Timestamp:    $TIMESTAMP"
echo "Results Dir:  $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Function to run a test
run_test() {
  local test_name=$1
  local test_file="${SCRIPT_DIR}/${test_name}.js"
  
  if [ ! -f "$test_file" ]; then
    echo "❌ Test file not found: $test_file"
    return 1
  fi
  
  echo "▶️  Running: $test_name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  local output_file="${RESULTS_DIR}/${test_name}-${TIMESTAMP}.json"
  local summary_file="${RESULTS_DIR}/${test_name}-${TIMESTAMP}-summary.json"
  
  k6 run "$test_file" \
    -e BASE_URL="$BASE_URL" \
    -e AUTH_TOKEN="$AUTH_TOKEN" \
    --out json="$output_file" \
    --summary-export="$summary_file"
  
  if [ $? -eq 0 ]; then
    echo "✅ PASSED: $test_name"
  else
    echo "❌ FAILED: $test_name"
    return 1
  fi
  
  echo ""
}

# Run tests based on type
case $TEST_TYPE in
  all)
    echo "Running all performance tests..."
    echo ""
    run_test "queue-operations" || true
    run_test "alert-operations" || true
    run_test "snapshot-operations" || true
    ;;
  queue)
    echo "Running queue operations tests..."
    echo ""
    run_test "queue-operations"
    ;;
  alert)
    echo "Running alert operations tests..."
    echo ""
    run_test "alert-operations"
    ;;
  snapshot)
    echo "Running snapshot operations tests..."
    echo ""
    run_test "snapshot-operations"
    ;;
  *)
    echo "❌ Unknown test type: $TEST_TYPE"
    echo "Valid options: all, queue, alert, snapshot"
    exit 1
    ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PERFORMANCE TESTING COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results saved to: $RESULTS_DIR"
echo ""
echo "View results with:"
echo "  - JSON output: cat $RESULTS_DIR/*-${TIMESTAMP}.json"
echo "  - Summary:     cat $RESULTS_DIR/*-${TIMESTAMP}-summary.json"
echo "  - k6 Cloud:    k6 cloud <test-file>"
echo ""
echo "To analyze results:"
echo "  - Import JSON to Excel/Grafana for visualization"
echo "  - Compare against baseline for regression detection"
echo "  - Share summary with team for capacity planning"
echo ""
