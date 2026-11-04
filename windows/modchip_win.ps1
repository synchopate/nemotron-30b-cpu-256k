param(
    [string]$Mode = "Menu",
    [int]$Profile = 0,
    [int]$Port = 8080,
    [int]$Ctx = 0,
    [int]$CpuMoe = -1,
    [int]$MoeTopK = 6,
    [double]$MoeMinW = 0.0
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BinDir    = Join-Path $ScriptDir "bin"
$ModelDir  = Join-Path $ScriptDir "models"

if (-not (Test-Path $BinDir))   { New-Item -ItemType Directory -Path $BinDir   | Out-Null }
if (-not (Test-Path $ModelDir)) { New-Item -ItemType Directory -Path $ModelDir | Out-Null }

function Write-Banner {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  llama-modchip v2.4.0 -- Windows Hybrid GPU + CPU MoE Edition" -ForegroundColor Green
    Write-Host "  Ottimizzato per NVIDIA RTX / AMD Vulkan (8-16 GB VRAM) + CPU (16-128 GB RAM)" -ForegroundColor White
    Write-Host "================================================================================" -ForegroundColor Cyan
}

function Get-HardwareInfo {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $os  = Get-CimInstance Win32_OperatingSystem
    $cpuName = ($cpu.Name -replace '\s+', ' ').Trim()
    $physCores = [int]$cpu.NumberOfCores
    $logCores  = [int]$cpu.NumberOfLogicalProcessors
    if ($physCores -le 0) { $physCores = 8 }

    $totalRamGB = [math]::Round($os.TotalVisibleMemorySize / 1048576, 1)
    $freeRamGB  = [math]::Round($os.FreePhysicalMemory / 1048576, 1)

    $gpuName = "Nessuna GPU NVIDIA rilevata (Fallback Vulkan/CPU)"
    $vramTotalMB = 0
    $vramFreeMB  = 0
    $hasNvidia   = $false

    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi -and (Test-Path "C:\Windows\System32\nvidia-smi.exe")) {
        $nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
    }

    if ($nvidiaSmi) {
        try {
            $smiOut = & $nvidiaSmi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            if ($smiOut) {
                $parts = $smiOut.Split(",")
                $gpuName     = $parts[0].Trim()
                $vramTotalMB = [int]$parts[1].Trim()
                $vramFreeMB  = [int]$parts[2].Trim()
                $hasNvidia   = $true
            }
        } catch {}
    }

    $driveLetter = (Split-Path -Qualifier $ScriptDir).TrimEnd(':')
    $disk = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
    $freeDiskGB = if ($disk) { [math]::Round($disk.Free / 1GB, 1) } else { 100.0 }

    return [PSCustomObject]@{
        CpuName     = $cpuName
        PhysCores   = $physCores
        LogCores    = $logCores
        TotalRamGB  = $totalRamGB
        FreeRamGB   = $freeRamGB
        HasNvidia   = $hasNvidia
        GpuName     = $gpuName
        VramTotalMB = $vramTotalMB
        VramFreeMB  = $vramFreeMB
        FreeDiskGB  = $freeDiskGB
    }
}

function Show-HardwareSummary($hw) {
    Write-Host ""
    Write-Host "  [PROFILO HARDWARE RILEVATO]" -ForegroundColor Yellow
    Write-Host ("  - CPU    : {0} ({1} core fisici / {2} thread)" -f $hw.CpuName, $hw.PhysCores, $hw.LogCores)
    if ($hw.HasNvidia) {
        $vTot = [math]::Round($hw.VramTotalMB / 1024, 1)
        $vFree = [math]::Round($hw.VramFreeMB / 1024, 1)
        Write-Host ("  - GPU    : {0} ({1} GB VRAM totali, {2} GB VRAM liberi)" -f $hw.GpuName, $vTot, $vFree) -ForegroundColor Green
    } else {
        Write-Host ("  - GPU    : {0}" -f $hw.GpuName) -ForegroundColor Yellow
    }
    $ramColor = if ($hw.FreeRamGB -lt 8.5) { "Yellow" } else { "Green" }
    Write-Host ("  - RAM    : {0} GB totali ({1} GB liberi in questo momento)" -f $hw.TotalRamGB, $hw.FreeRamGB) -ForegroundColor $ramColor
    Write-Host ("  - Disco  : {0} GB liberi su {1}" -f $hw.FreeDiskGB, (Split-Path -Qualifier $ScriptDir))
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
}

function Invoke-ResumableDownload([string]$Url, [string]$DestPath, [string]$Label) {
    Write-Host ("`n[*] Download: {0}" -f $Label) -ForegroundColor Cyan
    Write-Host ("    URL : {0}" -f $Url) -ForegroundColor DarkGray
    Write-Host ("    File: {0}" -f $DestPath) -ForegroundColor DarkGray
    Write-Host "    (Se il download si interrompe, rilancia pure lo script: riprendera' da dove era rimasto!)" -ForegroundColor Yellow

    $curlExe = "C:\Windows\System32\curl.exe"
    if (Test-Path $curlExe) {
        & $curlExe -L --fail --retry 5 --retry-delay 3 -C - -o $DestPath $Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download fallito con curl.exe (codice $LASTEXITCODE)."
        }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $DestPath -UseBasicParsing
    }
}

function Expand-ZipFast([string]$ZipPath, [string]$DestDir) {
    Write-Host ("[*] Estrazione archivio {0} ..." -f (Split-Path -Leaf $ZipPath)) -ForegroundColor Cyan
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $archive.Entries) {
            if ([string]::IsNullOrEmpty($entry.Name)) { continue }
            $targetFile = Join-Path $DestDir $entry.Name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetFile, $true)
        }
    } finally {
        $archive.Dispose()
    }
}

function Ensure-LlamaBinaries($hw) {
    $serverExe = Join-Path $BinDir "llama-server.exe"
    $cudaDll   = Join-Path $BinDir "ggml-cuda.dll"
    if ((Test-Path $serverExe) -and ((-not $hw.HasNvidia) -or (Test-Path $cudaDll))) {
        return $serverExe
    }

    Write-Host "`n[*] Prima esecuzione: scarico il motore llama.cpp ufficiale con accelerazione GPU..." -ForegroundColor Green
    $headers = @{ "User-Agent" = "llama-modchip-windows/2.3" }
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5" -Headers $headers

    $chosenRel = $null
    $binAsset  = $null
    $rtAsset   = $null

    foreach ($rel in $releases) {
        if ($hw.HasNvidia) {
            $b = $rel.assets | Where-Object { $_.name -match "^llama-b\d+-bin-win-cuda-12\.\d+-x64\.zip$" } | Select-Object -First 1
            $r = $rel.assets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$" } | Select-Object -First 1
            if ($b -and $r) {
                $chosenRel = $rel
                $binAsset  = $b
                $rtAsset   = $r
                break
            }
        } else {
            $b = $rel.assets | Where-Object { $_.name -match "^llama-b\d+-bin-win-vulkan-x64\.zip$" } | Select-Object -First 1
            if ($b) {
                $chosenRel = $rel
                $binAsset  = $b
                break
            }
        }
    }

    if (-not $binAsset) {
        throw "Impossibile trovare i binari Windows nelle ultime release di ggml-org/llama.cpp."
    }

    Write-Host ("[*] Versione motore selezionata: llama.cpp {0}" -f $chosenRel.tag_name) -ForegroundColor Green

    if ($rtAsset) {
        $rtZip = Join-Path $BinDir $rtAsset.name
        if (-not (Test-Path $rtZip)) {
            Invoke-ResumableDownload -Url $rtAsset.browser_download_url -DestPath $rtZip -Label ("CUDA 12.4 Runtime DLLs ({0} MB)" -f [math]::Round($rtAsset.size / 1MB))
        }
        Expand-ZipFast -ZipPath $rtZip -DestDir $BinDir
    }

    $binZip = Join-Path $BinDir $binAsset.name
    if (-not (Test-Path $binZip)) {
        Invoke-ResumableDownload -Url $binAsset.browser_download_url -DestPath $binZip -Label ("llama.cpp CUDA/GPU Engine ({0} MB)" -f [math]::Round($binAsset.size / 1MB))
    }
    Expand-ZipFast -ZipPath $binZip -DestDir $BinDir

    if (-not (Test-Path $serverExe)) {
        throw "llama-server.exe non trovato dopo l'estrazione in $BinDir"
    }
    Write-Host "[OK] Motore GPU installato correttamente in bin\!" -ForegroundColor Green
    return $serverExe
}

$ModelCatalog = @(
    [PSCustomObject]@{
        Id          = 1
        Title       = "[CONSIGLIATO 16GB RAM] Qwen3-30B-A3B (Q3_K_M, 13.7 GiB) -- 30 Miliardi Sparse MoE"
        Subtitle    = "Split Ibrido GPU+CPU: ~9.2 GB sulla RTX 2080 Ti + ~4.5 GB in RAM. Lascia Windows leggerissimo!"
        FileName    = "Qwen3-30B-A3B-Q3_K_M.gguf"
        SizeBytes   = 14711847488
        SizeGB      = 13.7
        Url         = "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q3_K_M.gguf"
        Kind        = "qwen3_moe"
        TotalMoe    = 48
        MoeLayerMB  = 265
        BaseVramMB  = 2100
        DefaultCtx  = 32768
    },
    [PSCustomObject]@{
        Id          = 2
        Title       = "[LONG-CONTEXT 128K-256K] NVIDIA-Nemotron-3.5-30B-A3B (Q4_0, 17.6 GiB) -- 30B Mamba-2 + MoE"
        Subtitle    = "Split Ibrido GPU+CPU: ~9.6 GB sulla RTX 2080 Ti + ~8.0 GB in RAM (chiudi il browser prima!)."
        FileName    = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
        SizeBytes   = 18898091584
        SizeGB      = 17.6
        Url         = "https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/8cd082170ed313484be1cf44f256a5caa2cfdad3/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
        Kind        = "nemotron_moe"
        TotalMoe    = 23
        MoeLayerMB  = 585
        BaseVramMB  = 4750
        DefaultCtx  = 131072
    },
    [PSCustomObject]@{
        Id          = 3
        Title       = "[100% GPU RTX 2080 Ti] Qwen3-14B (Q4_K_M, 8.4 GiB) -- 14 Miliardi Denso (0 GB RAM usata)"
        Subtitle    = "Entra TUTTO negli 11 GB della RTX 2080 Ti (616 GB/s)! Velocita' ~50-58 tok/s, zero carico RAM."
        FileName    = "Qwen3-14B-Q4_K_M.gguf"
        SizeBytes   = 9001752960
        SizeGB      = 8.4
        Url         = "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
        Kind        = "dense_gpu"
        TotalMoe    = 0
        MoeLayerMB  = 0
        BaseVramMB  = 9200
        DefaultCtx  = 32768
    },
    [PSCustomObject]@{
        Id          = 4
        Title       = "[100% GPU TURBO 128K] NVIDIA-Nemotron-H-8B-Reasoning (Q8_0, 8.0 GiB) -- 100+ tok/s!"
        Subtitle    = "Precisione 8-bit nativa, 128.000 token di contesto in soli 210 MB KV, 100% sulla RTX 2080 Ti!"
        FileName    = "nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf"
        SizeBytes   = 8619887584
        SizeGB      = 8.0
        Url         = "https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF/resolve/main/nvidia_Nemotron-H-8B-Reasoning-128K-Q8_0.gguf"
        Kind        = "dense_gpu"
        TotalMoe    = 0
        MoeLayerMB  = 0
        BaseVramMB  = 8600
        DefaultCtx  = 131072
    }
)

function Ensure-ModelFile($modelObj, $hw) {
    $dest = Join-Path $ModelDir $modelObj.FileName
    if (Test-Path $dest) {
        $len = (Get-Item $dest).Length
        if ($len -ge ($modelObj.SizeBytes * 0.98)) {
            Write-Host ("[OK] Modello gia' presente: {0} ({1} GiB)" -f $modelObj.FileName, [math]::Round($len / 1GB, 2)) -ForegroundColor Green
            return $dest
        }
    }

    $needGB = [math]::Round($modelObj.SizeGB + 1.0, 1)
    if ($hw.FreeDiskGB -lt $needGB) {
        throw ("Spazio su disco insufficiente: servono {0} GB liberi, disponibili {1} GB." -f $needGB, $hw.FreeDiskGB)
    }

    Invoke-ResumableDownload -Url $modelObj.Url -DestPath $dest -Label ("{0} ({1} GiB)" -f $modelObj.FileName, $modelObj.SizeGB)
    return $dest
}

function Compute-OptimalCpuMoe($modelObj, $hw, [int]$ExplicitCpuMoe) {
    if ($ExplicitCpuMoe -ge 0) { return $ExplicitCpuMoe }
    if ($modelObj.TotalMoe -le 0) { return 0 }
    if (-not $hw.HasNvidia) { return $modelObj.TotalMoe }

    # Rileva la VRAM libera aggiornata al secondo prima del lancio
    $freshHw = Get-HardwareInfo
    $freeVram = $freshHw.VramFreeMB
    if ($freeVram -le 0) { $freeVram = $hw.VramTotalMB - 900 }

    # Lascia ~550 MB di margine WDDM a Windows per non far mai traboccare la 2080 Ti nella RAM condivisa (GTT)
    $safetyMarginMB = 550
    $usableForMoeMB = $freeVram - $modelObj.BaseVramMB - $safetyMarginMB
    if ($usableForMoeMB -lt 0) { $usableForMoeMB = 0 }

    $gpuMoeLayers = [math]::Floor($usableForMoeMB / $modelObj.MoeLayerMB)
    if ($gpuMoeLayers -gt $modelObj.TotalMoe) { $gpuMoeLayers = $modelObj.TotalMoe }
    if ($gpuMoeLayers -lt 0) { $gpuMoeLayers = 0 }

    $cpuMoeLayers = $modelObj.TotalMoe - [int]$gpuMoeLayers
    return $cpuMoeLayers
}

function Run-QuickBenchmark($serverExe, $modelPath, $modelObj, $hw) {
    $benchExe = Join-Path $BinDir "llama-bench.exe"
    if (-not (Test-Path $benchExe)) {
        Write-Host "[!] llama-bench.exe non trovato in bin\." -ForegroundColor Red
        return
    }
    $nCpuMoe = Compute-OptimalCpuMoe -modelObj $modelObj -hw $hw -ExplicitCpuMoe $CpuMoe
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "  ESECUZIONE BENCHMARK HARDWARE (Prefill pp512 + Decode tg128)" -ForegroundColor Yellow
    Write-Host ("  Modello  : {0}" -f (Split-Path -Leaf $modelPath))
    Write-Host ("  Split    : -ngl 99 --n-cpu-moe {0} | Thread CPU: -t {1}" -f $nCpuMoe, $hw.PhysCores)
    Write-Host "================================================================================" -ForegroundColor Cyan

    $benchArgs = @(
        "-m", $modelPath,
        "-ngl", "99",
        "-t", "$($hw.PhysCores)",
        "-fa", "1",
        "-ctk", "q8_0",
        "-ctv", "q4_0",
        "-p", "512",
        "-n", "128",
        "-r", "2"
    )
    if ($nCpuMoe -gt 0) {
        $benchArgs += @("-ncmoe", "$nCpuMoe")
    }
    & $benchExe @benchArgs
}

# ==============================================================================
# MAIN FLOW
# ==============================================================================
Write-Banner
$hw = Get-HardwareInfo
Show-HardwareSummary $hw
$serverExe = Ensure-LlamaBinaries $hw

if ($Mode -eq "Bench") {
    $existing = Get-ChildItem -Path $ModelDir -Filter "*.gguf" | Sort-Object Length -Descending | Select-Object -First 1
    if (-not $existing) {
        Write-Host "`n[*] Nessun modello ancora scaricato. Scarico il profilo consigliato [1] per il benchmark..." -ForegroundColor Yellow
        $modelPath = Ensure-ModelFile -modelObj $ModelCatalog[0] -hw $hw
        $mObj = $ModelCatalog[0]
    } else {
        $modelPath = $existing.FullName
        $mObj = $ModelCatalog | Where-Object { $_.FileName -eq $existing.Name } | Select-Object -First 1
        if (-not $mObj) { $mObj = $ModelCatalog[0] }
    }
    Run-QuickBenchmark -serverExe $serverExe -modelPath $modelPath -modelObj $mObj -hw $hw
    exit 0
}

if ($Profile -le 0) {
    Write-Host "`n  SCEGLI IL MODELLO DA AVVIARE (premi Invio per [1] Consigliato):" -ForegroundColor Cyan
    Write-Host ""
    foreach ($m in $ModelCatalog) {
        $installedPath = Join-Path $ModelDir $m.FileName
        $badge = if (Test-Path $installedPath) { "[GIA' SCARICATO]" } else { "[Download auto]" }
        $color = if ($m.Id -eq 1) { "Green" } else { "White" }
        Write-Host ("  [{0}] {1} {2}" -f $m.Id, $m.Title, $badge) -ForegroundColor $color
        Write-Host ("      -> {0}" -f $m.Subtitle) -ForegroundColor DarkGray
        Write-Host ""
    }
    Write-Host "  [5] Usa un altro file .gguf presente nella cartella models\" -ForegroundColor White
    Write-Host "  [6] Esegui Benchmark di velocita' (llama-bench) sul modello installato" -ForegroundColor Yellow
    Write-Host ""
    $ans = Read-Host "  Seleziona [1-6] (Default = 1)"
    if ([string]::IsNullOrWhiteSpace($ans)) { $Profile = 1 } else { $Profile = [int]$ans }
}

if ($Profile -eq 6) {
    $existing = Get-ChildItem -Path $ModelDir -Filter "*.gguf" | Sort-Object Length -Descending | Select-Object -First 1
    if (-not $existing) {
        $modelPath = Ensure-ModelFile -modelObj $ModelCatalog[0] -hw $hw
        $mObj = $ModelCatalog[0]
    } else {
        $modelPath = $existing.FullName
        $mObj = $ModelCatalog | Where-Object { $_.FileName -eq $existing.Name } | Select-Object -First 1
        if (-not $mObj) { $mObj = $ModelCatalog[0] }
    }
    Run-QuickBenchmark -serverExe $serverExe -modelPath $modelPath -modelObj $mObj -hw $hw
    exit 0
}

$selectedModel = $null
$modelPath     = $null

if ($Profile -eq 5) {
    $customFiles = @(Get-ChildItem -Path $ModelDir -Filter "*.gguf")
    if ($customFiles.Count -eq 0) {
        throw "Nessun file .gguf trovato in $ModelDir. Copia li' il tuo .gguf o scegli [1-4]."
    }
    Write-Host "`n  File .gguf trovati in models\:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $customFiles.Count; $i++) {
        $sz = [math]::Round($customFiles[$i].Length / 1GB, 2)
        Write-Host ("    [{0}] {1} ({2} GiB)" -f ($i + 1), $customFiles[$i].Name, $sz)
    }
    $cIdx = Read-Host "  Quale file vuoi avviare? (Default = 1)"
    $pick = if ([string]::IsNullOrWhiteSpace($cIdx)) { 0 } else { ([int]$cIdx) - 1 }
    $modelPath = $customFiles[$pick].FullName
    $fileGB = $customFiles[$pick].Length / 1GB
    $selectedModel = [PSCustomObject]@{
        FileName   = $customFiles[$pick].Name
        TotalMoe   = if ($fileGB -gt 11.0) { 23 } else { 0 }
        MoeLayerMB = 500
        BaseVramMB = 4200
        DefaultCtx = 65536
    }
} else {
    $selectedModel = $ModelCatalog | Where-Object { $_.Id -eq $Profile } | Select-Object -First 1
    if (-not $selectedModel) { $selectedModel = $ModelCatalog[0] }
    $modelPath = Ensure-ModelFile -modelObj $selectedModel -hw $hw
}

$nCpuMoe = Compute-OptimalCpuMoe -modelObj $selectedModel -hw $hw -ExplicitCpuMoe $CpuMoe
$ctxSize = if ($Ctx -gt 0) { $Ctx } else { $selectedModel.DefaultCtx }

# Controllo preventivo RAM libera per modelli 30B su PC con 16 GB di RAM
$freshHw = Get-HardwareInfo
$estCpuRamGB = [math]::Round(($nCpuMoe * $selectedModel.MoeLayerMB) / 1024.0 + 0.8, 1)
if (($nCpuMoe -gt 0) -and ($freshHw.FreeRamGB -lt ($estCpuRamGB + 0.5))) {
    Write-Host ""
    Write-Host ("  [!] ATTENZIONE RAM: I {0} layer MoE su CPU richiedono ~{1} GB di RAM," -f $nCpuMoe, $estCpuRamGB) -ForegroundColor Yellow
    Write-Host ("      ma Windows ha attualmente solo {0} GB di RAM libera (su {1} GB totali)." -f $freshHw.FreeRamGB, $freshHw.TotalRamGB) -ForegroundColor Yellow
    Write-Host "      Consiglio: chiudi Chrome/browser pesanti prima di continuare per evitare swap su SSD!" -ForegroundColor Yellow
    if ($ctxSize -gt 65536) {
        $ctxSize = 65536
        Write-Host "      -> Contesto ridotto automaticamente a 65.536 token (64K) per risparmiare RAM." -ForegroundColor Cyan
    }
    Read-Host "      Premi INVIO quando sei pronto per avviare il motore" | Out-Null
}

$helpOut = & $serverExe --help 2>&1 | Out-String
$loadModeArgs = @()
if ($helpOut -match "--load-mode") {
    $loadModeArgs = @("--load-mode", "none")
} elseif ($helpOut -match "--no-mmap") {
    $loadModeArgs = @("--no-mmap")
}

$serverArgs = @(
    "-m", $modelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-c", "$ctxSize",
    "-ngl", "99"
) + $loadModeArgs + @(
    "-ctk", "q8_0",
    "-ctv", "q4_0",
    "-fa", "on",
    "-t", "$($hw.PhysCores)",
    "-b", "2048",
    "-ub", "512",
    "--cache-reuse", "256",
    "--spec-type", "ngram-simple",
    "--spec-ngram-size-n", "3",
    "--spec-ngram-size-m", "5",
    "--draft-max", "6",
    "--reasoning-format", "deepseek"
)

if ($nCpuMoe -gt 0) {
    $serverArgs += @("--n-cpu-moe", "$nCpuMoe")
}

# Configura il controllo MoE runtime L2-quadrature (%TEMP%\dsv4_moe_cfg.bin + variabili ambiente)
if ($MoeTopK -lt 6 -or $MoeMinW -gt 0.0) {
    $env:DSV4_MOE_TOPK = "$MoeTopK"
    $env:DSV4_MOE_MIN_W = "$MoeMinW"
    $env:LLAMA_MOE_TOP_K = "$MoeTopK"
}
try {
    $cfgPath = Join-Path ([System.IO.Path]::GetTempPath()) "dsv4_moe_cfg.bin"
    $ms = New-Object System.IO.MemoryStream
    $bw = New-Object System.IO.BinaryWriter($ms)
    $bw.Write([int32]$MoeTopK)
    $bw.Write([int32]$MoeTopK)
    $bw.Write([int32]([math]::Round($MoeMinW * 10000)))
    $bw.Write([int32]3)
    $bw.Write([int32]43)
    $bw.Write([int32]1) # L2 quadrature norm-preserving rescaling
    $bw.Write([int32]0)
    $bw.Write([int32]0)
    $bw.Flush()
    [System.IO.File]::WriteAllBytes($cfgPath, $ms.ToArray())
    $bw.Close()
} catch {}

$gpuMoeCount = $selectedModel.TotalMoe - $nCpuMoe
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  AVVIO MOTORE IBRIDO GPU + CPU (llama-modchip Windows)" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ("  - Modello attivo     : {0}" -f $selectedModel.FileName) -ForegroundColor White
Write-Host ("  - Contesto attivo    : {0} token (KV Cache quantizzata Q8_0/Q4_0 + Flash-Attn)" -f $ctxSize) -ForegroundColor White
if ($selectedModel.TotalMoe -gt 0) {
    Write-Host ("  - Tensor-Role Split  : Backbone + {0}/{1} layer MoE su GPU ({2})" -f $gpuMoeCount, $selectedModel.TotalMoe, $hw.GpuName) -ForegroundColor Cyan
    Write-Host ("                         {0}/{1} layer MoE su CPU RAM (~{2} GB con --no-mmap)" -f $nCpuMoe, $selectedModel.TotalMoe, $estCpuRamGB) -ForegroundColor Cyan
} else {
    Write-Host ("  - Offload GPU        : 100% su {0} (-ngl 99, 0 GB RAM di sistema)" -f $hw.GpuName) -ForegroundColor Cyan
}
Write-Host ("  - Thread CPU Fisici  : -t {0} (Evita HyperThreading per non saturare il bus DDR4)" -f $hw.PhysCores) -ForegroundColor White
Write-Host ("  - Speculative Engine : Draftless N-Gram Cascade (m=5, draft-max=6)" -f $hw.PhysCores) -ForegroundColor White
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ("  -> Web Chat UI       : http://127.0.0.1:{0}  (Si aprira' nel browser appena pronto!)" -f $Port) -ForegroundColor Yellow
Write-Host ("  -> OpenAI API Locale : http://127.0.0.1:{0}/v1 (Per VS Code Continue / Cline / Cursor)" -f $Port) -ForegroundColor Yellow
Write-Host "  (Premi CTRL+C in questa finestra quando vuoi spegnere il server)" -ForegroundColor DarkGray
Write-Host "================================================================================`n" -ForegroundColor Green

# Avvia un job leggero in background che attende /health e apre il browser automaticamente
$healthUrl = "http://127.0.0.1:$Port/health"
$chatUrl   = "http://127.0.0.1:$Port"
$null = Start-Job -ScriptBlock {
    param($hUrl, $cUrl)
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -Uri $hUrl -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            if ($r.StatusCode -eq 200) {
                Start-Process $cUrl
                break
            }
        } catch {}
    }
} -ArgumentList $healthUrl, $chatUrl

& $serverExe @serverArgs
