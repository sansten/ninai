# Performance Testing Suite Runner (Windows PowerShell)
#
# This script runs all performance tests and generates a comprehensive report.
# It can be used for:
# - Local development baseline testing
# - Pre-deployment validation
# - Capacity planning
# - Regression detection
#
# Usage:
#   .\run-performance-tests.ps1 -Environment <env> -TestType <type>
#
# Examples:
#   .\run-performance-tests.ps1                                        # Run all tests against localhost
#   .\run-performance-tests.ps1 -Environment staging                   # Run all tests against staging
#   .\run-performance-tests.ps1 -Environment prod -TestType queue      # Run only queue tests against prod
#   .\run-performance-tests.ps1 -Environment local -TestType all       # Run all tests

param(
    [string]$Environment = "local",
    [string]$TestType = "all",
    [string]$AuthToken = $env:AUTH_TOKEN
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommandPath
$ResultsDir = Join-Path $ScriptDir "results"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

# Environment mappings
switch ($Environment) {
    "local" {
        $BaseUrl = "http://localhost:8000"
        Write-Host "📊 Running tests against LOCAL (http://localhost:8000)" -ForegroundColor Green
    }
    "staging" {
        $BaseUrl = "https://api-staging.ninai.com"
        Write-Host "📊 Running tests against STAGING" -ForegroundColor Green
    }
    "prod" {
        $BaseUrl = "https://api.ninai.com"
        Write-Host "⚠️  WARNING: Running tests against PRODUCTION" -ForegroundColor Red
        Write-Host "⚠️  Make sure you understand the impact of load tests!" -ForegroundColor Red
        $confirm = Read-Host "Press ENTER to continue or type 'cancel' to abort"
        if ($confirm -eq "cancel") {
            exit
        }
    }
    default {
        Write-Host "❌ Unknown environment: $Environment" -ForegroundColor Red
        Write-Host "Valid options: local, staging, prod" -ForegroundColor Red
        exit 1
    }
}

# Check auth token
if (-not $AuthToken) {
    Write-Host "❌ AUTH_TOKEN environment variable not set" -ForegroundColor Red
    Write-Host "Set it with: `$env:AUTH_TOKEN = 'your-token'" -ForegroundColor Red
    exit 1
}

# Create results directory
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  NINAI PERFORMANCE TESTING SUITE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Environment:  $Environment"
Write-Host "Base URL:     $BaseUrl"
Write-Host "Test Type:    $TestType"
Write-Host "Timestamp:    $Timestamp"
Write-Host "Results Dir:  $ResultsDir"
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Function to run a test
function Run-Test {
    param(
        [string]$TestName
    )
    
    $TestFile = Join-Path $ScriptDir "$TestName.js"
    
    if (-not (Test-Path $TestFile)) {
        Write-Host "❌ Test file not found: $TestFile" -ForegroundColor Red
        return $false
    }
    
    Write-Host "▶️  Running: $TestName" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    $OutputFile = Join-Path $ResultsDir "$TestName-$Timestamp.json"
    $SummaryFile = Join-Path $ResultsDir "$TestName-$Timestamp-summary.json"
    
    # Run k6 test
    & k6 run $TestFile `
        -e BASE_URL=$BaseUrl `
        -e AUTH_TOKEN=$AuthToken `
        --out json=$OutputFile `
        --summary-export=$SummaryFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PASSED: $TestName" -ForegroundColor Green
    } else {
        Write-Host "❌ FAILED: $TestName" -ForegroundColor Red
        return $false
    }
    
    Write-Host ""
    return $true
}

# Run tests based on type
switch ($TestType) {
    "all" {
        Write-Host "Running all performance tests..." -ForegroundColor Cyan
        Write-Host ""
        Run-Test "queue-operations" | Out-Null
        Run-Test "alert-operations" | Out-Null
        Run-Test "snapshot-operations" | Out-Null
    }
    "queue" {
        Write-Host "Running queue operations tests..." -ForegroundColor Cyan
        Write-Host ""
        Run-Test "queue-operations" | Out-Null
    }
    "alert" {
        Write-Host "Running alert operations tests..." -ForegroundColor Cyan
        Write-Host ""
        Run-Test "alert-operations" | Out-Null
    }
    "snapshot" {
        Write-Host "Running snapshot operations tests..." -ForegroundColor Cyan
        Write-Host ""
        Run-Test "snapshot-operations" | Out-Null
    }
    default {
        Write-Host "❌ Unknown test type: $TestType" -ForegroundColor Red
        Write-Host "Valid options: all, queue, alert, snapshot" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PERFORMANCE TESTING COMPLETE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Results saved to: $ResultsDir" -ForegroundColor Green
Write-Host ""
Write-Host "View results with:"
Write-Host "  - JSON output: Get-Content '$ResultsDir\*-$Timestamp.json'"
Write-Host "  - Summary:     Get-Content '$ResultsDir\*-$Timestamp-summary.json'"
Write-Host "  - k6 Cloud:    k6 cloud <test-file>"
Write-Host ""
Write-Host "To analyze results:"
Write-Host "  - Import JSON to Excel/Grafana for visualization"
Write-Host "  - Compare against baseline for regression detection"
Write-Host "  - Share summary with team for capacity planning"
Write-Host ""
