$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$cpuPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$gpuPython = Join-Path $projectRoot '.venv-gpu\Scripts\python.exe'
$ready = Join-Path $projectRoot '.venv-gpu\domino-ready'
if (-not (Test-Path -LiteralPath $cpuPython)) {
    throw 'Install the CPU environment from README.md first.'
}
if (Test-Path -LiteralPath $ready) { Remove-Item -LiteralPath $ready }
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $gpuPython)) {
        & $cpuPython -m venv .venv-gpu
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create GPU environment.' }
    }
    & $gpuPython -m pip install -e '.[test]'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install application.' }
    & $gpuPython -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
    if ($LASTEXITCODE -ne 0) { throw 'Cannot replace ONNX Runtime.' }
    & $gpuPython -m pip install --requirement tools/requirements-gpu.txt
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install DirectML.' }
    & $gpuPython -m domino_video.ocr
    if ($LASTEXITCODE -ne 0) { throw 'GPU model check failed. CPU environment is unchanged.' }
    Set-Content -LiteralPath $ready -Value 'DirectML 1.24.4'
    Write-Output 'GPU environment is ready. Use run.bat; --device cpu forces CPU.'
} finally {
    Pop-Location
}
