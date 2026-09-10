param([string]$FxcPath = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe')
$ErrorActionPreference = 'Stop'
foreach ($name in @('EdgeGradient', 'FlowDisplace')) {
    $binary = Join-Path $PSScriptRoot "obj\Release\net10.0-windows10.0.19041.0\$name.cso"
    $assembly = (& $FxcPath /dumpbin $binary | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect $binary" }
    foreach ($semantic in @('SCENE_POSITION', 'TEXCOORD')) {
        if ($assembly -notmatch $semantic) { throw "$name has no $semantic input" }
    }
    $layout = switch ($name) {
        'EdgeGradient' { @(@('detectionScale', 0), @('inputBounds', 16)) }
        'FlowDisplace' {
            @(
                @('strength', 0), @('turbulence', 4), @('turbulenceDetail', 8), @('noiseScale', 12),
                @('flowSpeed', 16), @('time', 20), @('dispersion', 24), @('iridescence', 28),
                @('lightAngle', 32), @('seed', 36), @('threshold', 40), @('contrast', 44),
                @('outputMode', 48), @('dispersionSteps', 52), @('phaseOffset', 56), @('tapCount', 60), @('inputBounds', 64),
                @('lightAndMask', 80), @('spectralTaps\[128\]', 96)
            )
        }
    }
    foreach ($entry in $layout) {
        $field, $offset = $entry
        if ($assembly -notmatch "float[1-4]?\s+$field;\s+// Offset:\s+$offset\s") {
            throw "$name constant $field is not at byte offset $offset"
        }
    }
    Write-Output "PASS: $name constant buffer layout"
    Write-Output "PASS: $name receives Direct2D coordinates"
}
