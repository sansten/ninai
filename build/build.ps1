Param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$book = Join-Path $root 'book'
$out = Join-Path $root 'build/out'
$tmpCh = Join-Path $root 'build/tmp/chapters_pdf'
New-Item -ItemType Directory -Force -Path $out | Out-Null

# Validate chapters
python (Join-Path $book 'tools/critic_validate.py')
if ($LASTEXITCODE -ne 0) { Write-Host 'Validator reported issues (placeholders may remain).' }

# Pre-render mermaid diagrams for PDF
python (Join-Path $book 'tools/render_mermaid.py')

# Build HTML and PDF from pre-rendered chapters (diagrams as SVG images)
$chapterHtml = Get-ChildItem $tmpCh -Filter 'ch*.md' | ForEach-Object { $_.FullName }
& pandoc --standalone --toc --toc-depth=2 --metadata-file (Join-Path $root 'pandoc/pandoc.yaml') -o (Join-Path $out 'book.html') (Join-Path $book 'index.md') $chapterHtml

& pandoc --pdf-engine=wkhtmltopdf --metadata-file (Join-Path $root 'pandoc/pandoc.yaml') -o (Join-Path $out 'book.pdf') (Join-Path $book 'index.md') $chapterHtml

Write-Host "Built: $(Join-Path $out 'book.html') and $(Join-Path $out 'book.pdf')"