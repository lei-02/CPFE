$ErrorActionPreference = "Stop"

$environment = Join-Path $PSScriptRoot "..\..\.fortran-reference"
$environment = [IO.Path]::GetFullPath($environment)
$mingw = Join-Path $environment "Library\mingw-w64\bin"
$libraryBin = Join-Path $environment "Library\bin"
$scripts = Join-Path $environment "Scripts"
$gfortran = Join-Path $mingw "gfortran.exe"
$build = Join-Path $PSScriptRoot "build"
$driver = Join-Path $build "case2_trace_driver.exe"
$env:Path = "$mingw;$libraryBin;$scripts;$environment;$env:Path"
$env:OXFORD_GFORTRAN_BIN = $mingw
$env:OXFORD_REFERENCE_DRIVER = $driver

if (-not (Test-Path -LiteralPath $gfortran)) {
    throw "GNU Fortran compiler was not found at $gfortran"
}
New-Item -ItemType Directory -Force -Path $build | Out-Null

& $gfortran `
    -O2 `
    "-J$build" `
    (Join-Path $PSScriptRoot "case2_reference.f90") `
    (Join-Path $PSScriptRoot "case2_trace_driver.f90") `
    -o $driver
if ($LASTEXITCODE -ne 0) {
    throw "Fortran reference compilation failed with exit code $LASTEXITCODE"
}

& (Join-Path $environment "python.exe") `
    (Join-Path $PSScriptRoot "generate_traces.py") `
    --output (Join-Path $PSScriptRoot "..\..\tests\fixtures\oxford_case2_trace.npz")
if ($LASTEXITCODE -ne 0) {
    throw "Oxford trace generation failed with exit code $LASTEXITCODE"
}
