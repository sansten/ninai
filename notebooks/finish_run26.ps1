# finish_run26.ps1 - Run this when locomo_run26.log shows final scores
# Usage: cd d:/Sansten/Projects/Ninai2/repos/ninai/notebooks ; .\finish_run26.ps1

$logPath = "$PSScriptRoot\locomo_run26.log"
$repoPath = "d:/Sansten/Projects/Ninai2/repos/ninai"
$planPath = "d:/Sansten/Projects/Ninai2/LOCOMO_RUN26_PLAN.md"

Write-Host "=== LoCoMo Run 26 Finish Script ===" -ForegroundColor Cyan

# 1. Show final log output
Write-Host "`n--- Final log (last 40 lines) ---" -ForegroundColor Yellow
Get-Content $logPath -Tail 40 -Encoding Unicode

# 2. Extract scores from log
Write-Host "`n--- Extracting scores ---" -ForegroundColor Yellow
$logContent = Get-Content $logPath -Encoding Unicode -Raw
$scoreLines = $logContent | Select-String -Pattern "(adversarial|multi_hop|single_hop|temporal|open_domain|overall|ROUGE)" -AllMatches

# 3. Git workflow
Write-Host "`n--- Git workflow ---" -ForegroundColor Yellow
Set-Location $repoPath

$status = git status --short
Write-Host "Git status: $status"

git add notebooks/locomo_run.py
git commit -m "benchmark: locomo run 26 -- revert adversarial prompt, gemma4 for open_domain, higher multi_hop limit"

# Check if branch already exists
$branchName = "benchmark/locomo-run26-tuning"
$existing = git branch --list $branchName
if ($existing) {
    Write-Host "Branch $branchName already exists, switching to it"
    git switch $branchName
} else {
    git switch -c $branchName
}

git push -u origin $branchName

# 4. Create PR
Write-Host "`n--- Creating PR ---" -ForegroundColor Yellow
$prBody = @"
## LoCoMo Run 26 - Three Fixes

### Changes
- **Fix 1**: Reverted adversarial prompt to Run 24 terse version (removed the 'If the question assumes a fact NOT in the conversation...' instruction)
- **Fix 2**: Route `open_domain` category to `gemma4:e4b` (split from deepseek; new model with better base for open-domain QA); deepseek now handles only `multi_hop`  
- **Fix 3**: Double retrieval limit for `multi_hop` (`min(RETRIEVAL_LIMIT * 2, 80)`) for richer context

### Model Routing (Run 26)
- `qwen2.5:7b`: adversarial, single_hop, temporal (1049 prompts)
- `gemma4:e4b` (new): open_domain (841 prompts, workers=6, ctx=16384)
- `deepseek-coder-v2:16b`: multi_hop only (96 prompts, workers=4, ctx=24576)

See LOCOMO_RUN26_PLAN.md for full benchmark results.
"@

$pr = gh pr create --title "benchmark: locomo run 26 -- revert adversarial prompt, gemma4 for open_domain, higher multi_hop limit" --body $prBody 2>&1
Write-Host "PR created: $pr"

# Extract PR number
$prNumber = ($pr | Select-String -Pattern '#(\d+)').Matches[0].Groups[1].Value
Write-Host "PR number: $prNumber"

# 5. Merge PR
Write-Host "`n--- Merging PR ---" -ForegroundColor Yellow
gh pr merge $prNumber --merge --admin --delete-branch

Write-Host "`n=== Run 26 Complete! ===" -ForegroundColor Green
