# Run the complete v16 MSE=5e-5 experiment unattended on Windows.
#
# The default profile is budgeted from measurements on this workspace's GTX 1070:
# training normally occupies about five hours and the paired benchmark gets the
# remaining wall-time budget.  An RTX 3090 will usually be faster, but no fixed
# speedup is assumed.  Every phase is recorded and an unqualified checkpoint is
# never labelled as a formal result.
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$RunName = "candidate_selection_v16_mse5e-5_k64",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [int]$TargetEpochs = 56,
    [int]$FallbackEpochs = 64,
    [double]$BudgetHours = 12.0,
    [int]$FreshProposalEpochs = 12,
    [int]$FreshBatchSize = 64,
    [int]$SyntheticSamplesPerK = 2,
    [int]$RealSamplesPerDataset = 8,
    [int]$KangAdmmIterations = 400,
    [int]$LuoDeIterations = 50,
    [string]$OutputRoot = "",
    [string]$InitCheckpoint = "outputs/checkpoints/candidate_selection_v16.proposal.pt",
    [switch]$SkipTraining,
    [switch]$Detach,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

if ($RunName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunName may contain only letters, digits, dot, underscore, and hyphen."
}
foreach ($positive in @{
    TargetEpochs = $TargetEpochs
    FallbackEpochs = $FallbackEpochs
    FreshProposalEpochs = $FreshProposalEpochs
    FreshBatchSize = $FreshBatchSize
    SyntheticSamplesPerK = $SyntheticSamplesPerK
    RealSamplesPerDataset = $RealSamplesPerDataset
    KangAdmmIterations = $KangAdmmIterations
    LuoDeIterations = $LuoDeIterations
}.GetEnumerator()) {
    if ($positive.Value -lt 1) {
        throw "$($positive.Key) must be positive."
    }
}
if ($FreshProposalEpochs -ge $TargetEpochs) {
    throw "FreshProposalEpochs must be smaller than TargetEpochs."
}
if ($FallbackEpochs -lt $TargetEpochs) {
    throw "FallbackEpochs must be greater than or equal to TargetEpochs."
}
if ([double]::IsNaN($BudgetHours) -or [double]::IsInfinity($BudgetHours) -or $BudgetHours -le 0) {
    throw "BudgetHours must be finite and positive."
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepositoryRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepositoryRoot "outputs"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$CheckpointDirectory = Join-Path $OutputRoot "checkpoints"
$LogDirectory = Join-Path (Join-Path $OutputRoot "logs") $RunName
$ComparisonRoot = Join-Path (Join-Path $OutputRoot "comparisons") $RunName
$FigureRoot = Join-Path (Join-Path $OutputRoot "figures") $RunName
foreach ($directory in @($CheckpointDirectory, $LogDirectory, $ComparisonRoot, $FigureRoot)) {
    [void](New-Item -ItemType Directory -Path $directory -Force)
}

$CheckpointPath = [System.IO.Path]::GetFullPath(
    (Join-Path $CheckpointDirectory ($RunName + ".pt"))
)
$LastPath = [System.IO.Path]::GetFullPath(
    (Join-Path $CheckpointDirectory ($RunName + ".last.pt"))
)
$ProposalPath = [System.IO.Path]::GetFullPath(
    (Join-Path $CheckpointDirectory ($RunName + ".proposal.pt"))
)
$HistoryPath = [System.IO.Path]::GetFullPath(
    (Join-Path $CheckpointDirectory ($RunName + ".history.json"))
)
$MasterLog = Join-Path $LogDirectory "overnight.log"
$ManifestPath = Join-Path $LogDirectory "overnight_manifest.json"

if ($Detach) {
    if ($DryRun) {
        throw "-Detach and -DryRun cannot be combined."
    }
    if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
        throw "-Detach is implemented for Windows PowerShell only."
    }
    function ConvertTo-PowerShellLiteral {
        param([object]$Value)
        return "'" + ([string]$Value).Replace("'", "''") + "'"
    }
    $ChildParts = New-Object System.Collections.Generic.List[string]
    $ChildParts.Add("& " + (ConvertTo-PowerShellLiteral $PSCommandPath))
    foreach ($Entry in @($PSBoundParameters.GetEnumerator() | Sort-Object Key)) {
        if ($Entry.Key -in @("Detach", "DryRun")) {
            continue
        }
        if ($Entry.Value -is [System.Management.Automation.SwitchParameter]) {
            if ([bool]$Entry.Value) {
                $ChildParts.Add("-" + $Entry.Key)
            }
            continue
        }
        $ChildParts.Add(
            ("-" + $Entry.Key + " " + (ConvertTo-PowerShellLiteral $Entry.Value))
        )
    }
    $Encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes(($ChildParts -join " "))
    )
    $PowerShellExecutable = (Get-Process -Id $PID).Path
    $LaunchStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $DetachedStdout = Join-Path $LogDirectory ("detached_" + $LaunchStamp + ".stdout.log")
    $DetachedStderr = Join-Path $LogDirectory ("detached_" + $LaunchStamp + ".stderr.log")
    $Child = Start-Process `
        -FilePath $PowerShellExecutable `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $Encoded) `
        -WorkingDirectory $RepositoryRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $DetachedStdout `
        -RedirectStandardError $DetachedStderr `
        -PassThru
    Write-Host "Detached overnight pipeline started as PID $($Child.Id)."
    Write-Host "Progress log: $MasterLog"
    Write-Host "Fallback stdout/stderr: $DetachedStdout / $DetachedStderr"
    exit 0
}

$RunState = [ordered]@{
    run_name = $RunName
    started_utc = [DateTime]::UtcNow.ToString("o")
    status = "initializing"
    requested_profile = [ordered]@{
        candidate_internal_knots = 64
        source_internal_knots = "4..24"
        source_control_points = "8..28"
        largest_source_full_knot_vector = 32
        train_size = 1500
        validation_size = 500
        mse_tolerance = 5e-5
        fresh_batch_size = $FreshBatchSize
        target_epochs = $TargetEpochs
        fallback_epochs = $FallbackEpochs
        budget_hours = $BudgetHours
        device = $Device
    }
    checkpoint = $null
    diagnostic_not_final = $null
    comparison_directory = $null
    figure_directory = $null
    phases = [ordered]@{}
}
$LockPath = Join-Path $LogDirectory "overnight.lock"
$LockStream = $null

function Save-RunState {
    $RunState | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
}

function Format-CommandToken {
    param([string]$Token)
    if ($Token -notmatch '[\s"]') {
        return $Token
    }
    return '"' + $Token.Replace('"', '\"') + '"'
}

function Invoke-LoggedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string[]]$CommandArguments,
        [switch]$AllowFailure
    )
    $PhaseLog = Join-Path $LogDirectory ($Phase + ".log")
    $Rendered = (@($Python) + $CommandArguments | ForEach-Object {
        Format-CommandToken ([string]$_)
    }) -join " "
    Write-Host ""
    Write-Host "[$Phase] $Rendered"
    if ($DryRun) {
        $RunState.phases[$Phase] = [ordered]@{
            status = "dry_run"
            command = $Rendered
        }
        Save-RunState
        return $true
    }
    $Started = Get-Date
    & $Python @CommandArguments 2>&1 |
        Tee-Object -FilePath $PhaseLog -Append |
        Out-Host
    $ExitCode = $LASTEXITCODE
    $RunState.phases[$Phase] = [ordered]@{
        status = $(if ($ExitCode -eq 0) { "completed" } else { "failed" })
        exit_code = $ExitCode
        seconds = ((Get-Date) - $Started).TotalSeconds
        command = $Rendered
        log = $PhaseLog
    }
    Save-RunState
    if ($ExitCode -ne 0) {
        $Message = "Phase '$Phase' failed with exit code $ExitCode. See $PhaseLog"
        if ($AllowFailure) {
            Write-Warning $Message
            return $false
        }
        throw $Message
    }
    return $true
}

function Get-V16TrainingProcesses {
    if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
        return @()
    }
    return @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                if ([string]::IsNullOrWhiteSpace($_.CommandLine)) {
                    return $false
                }
                $Normalized = $_.CommandLine.Replace('\', '/').ToLowerInvariant()
                return $Normalized.Contains('scripts/train_v16.py')
            }
    )
}

function Test-ProcessTargetsCheckpoint {
    param([Parameter(Mandatory = $true)]$Process)
    $Normalized = $Process.CommandLine.Replace('\', '/').ToLowerInvariant()
    $Relative = ("outputs/checkpoints/" + [System.IO.Path]::GetFileName($CheckpointPath)).ToLowerInvariant()
    $Absolute = $CheckpointPath.Replace('\', '/').ToLowerInvariant()
    $Name = [System.IO.Path]::GetFileName($CheckpointPath).ToLowerInvariant()
    return $Normalized.Contains($Absolute) -or
        $Normalized.Contains($Relative) -or
        [regex]::IsMatch(
            $Normalized,
            ('(^|[\s"'']{1})' + [regex]::Escape($Name) + '($|[\s"'']{1})')
        )
}

function Get-GpuComputeProcesses {
    if ($Device -ne "cuda") {
        return @()
    }
    $NvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($null -eq $NvidiaSmi) {
        throw "CUDA profile requested, but nvidia-smi is unavailable; GPU-conflict safety cannot be checked."
    }
    $Rows = & $NvidiaSmi.Source --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed; refusing to start an unattended CUDA job without a conflict check."
    }
    $Result = @()
    foreach ($Row in @($Rows)) {
        if ([string]::IsNullOrWhiteSpace($Row)) {
            continue
        }
        $Fields = $Row -split ',', 2
        $PidValue = 0
        if ([int]::TryParse($Fields[0].Trim(), [ref]$PidValue)) {
            $Result += [pscustomobject]@{
                ProcessId = $PidValue
                ProcessName = $(if ($Fields.Count -gt 1) { $Fields[1].Trim() } else { "unknown" })
            }
        }
    }
    return @($Result)
}

function Assert-NoForeignGpuWork {
    param([int[]]$AllowedProcessIds = @())
    if ($DryRun -or $Device -ne "cuda") {
        return
    }
    # Windows WDDM reports browser/desktop/overlay processes as compute clients
    # with N/A memory. They are not competing training jobs. Only another
    # Python CUDA workload is treated as an actionable conflict here; all
    # train_v16.py commands are checked separately from their command lines.
    $Foreign = @(
        Get-GpuComputeProcesses |
            Where-Object {
                $AllowedProcessIds -notcontains [int]$_.ProcessId -and
                $_.ProcessName -match '(?i)(^|[\\/])python(?:w)?(?:\.exe)?$' -and
                $null -ne (Get-Process -Id ([int]$_.ProcessId) -ErrorAction SilentlyContinue)
            }
    )
    if ($Foreign.Count -gt 0) {
        $Details = ($Foreign | ForEach-Object {
            "PID=$($_.ProcessId) process=$($_.ProcessName)"
        }) -join '; '
        throw "Another GPU compute process is active ($Details). No process was stopped; rerun when the GPU is free."
    }
}

function Wait-ForExistingTargetTraining {
    if ($DryRun) {
        Write-Host "[dry-run] Would detect and wait for an existing train_v16.py targeting $CheckpointPath."
        return
    }
    $AllTraining = @(Get-V16TrainingProcesses)
    $TargetTraining = @($AllTraining | Where-Object { Test-ProcessTargetsCheckpoint $_ })
    $ForeignTraining = @($AllTraining | Where-Object { -not (Test-ProcessTargetsCheckpoint $_) })
    if ($ForeignTraining.Count -gt 0) {
        $Details = ($ForeignTraining | ForEach-Object {
            "PID=$($_.ProcessId) command=$($_.CommandLine)"
        }) -join [Environment]::NewLine
        throw "A different train_v16.py job is already running. No process was stopped:`n$Details"
    }
    if ($TargetTraining.Count -gt 1) {
        throw "Multiple train_v16.py jobs target the same checkpoint; refusing to add another job."
    }
    $Allowed = @($TargetTraining | ForEach-Object { [int]$_.ProcessId })
    Assert-NoForeignGpuWork -AllowedProcessIds $Allowed
    if ($TargetTraining.Count -eq 0) {
        return
    }
    $Target = $TargetTraining[0]
    Write-Host "Existing matching training job detected: PID=$($Target.ProcessId). It will not be stopped."
    while ($null -ne (Get-Process -Id $Target.ProcessId -ErrorAction SilentlyContinue)) {
        $Now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$Now] Waiting for PID $($Target.ProcessId); next heartbeat in at most 45 seconds."
        try {
            Wait-Process -Id $Target.ProcessId -Timeout 45 -ErrorAction Stop
        } catch {
            if ($null -eq (Get-Process -Id $Target.ProcessId -ErrorAction SilentlyContinue)) {
                break
            }
        }
    }
    Write-Host "Existing matching training PID $($Target.ProcessId) has exited."
    # nvidia-smi may retain a just-exited WDDM row briefly; allow only that
    # exact PID here. The pre-launch check below runs again with no allowance.
    Assert-NoForeignGpuWork -AllowedProcessIds @([int]$Target.ProcessId)
}

function Get-CompletedEpoch {
    if (-not (Test-Path -LiteralPath $HistoryPath)) {
        return 0
    }
    try {
        $History = @(Get-Content -LiteralPath $HistoryPath -Raw -Encoding utf8 | ConvertFrom-Json)
        if ($History.Count -eq 0) {
            return 0
        }
        return [int](($History | Measure-Object -Property epoch -Maximum).Maximum)
    } catch {
        Write-Warning "Could not parse $HistoryPath; the last checkpoint will decide resume state."
        return 0
    }
}

function Get-ResumeArguments {
    param(
        [Parameter(Mandatory = $true)][string]$ResumeCheckpoint,
        [Parameter(Mandatory = $true)][int]$RequestedEpochs
    )
    $Reader = @'
import json
import sys
import torch
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(json.dumps(payload['training_config'], separators=(',', ':')))
'@
    $ConfigText = & $Python -c $Reader $ResumeCheckpoint
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read training_config from $ResumeCheckpoint"
    }
    $Config = $ConfigText | ConvertFrom-Json
    $Skip = @(
        "epochs", "resume", "init_checkpoint", "output", "device",
        "train_seed", "train_seed_stride"
    )
    $BooleanOptional = @("certified_minimal_source", "resample_train_each_epoch")
    $Arguments = New-Object System.Collections.Generic.List[string]
    foreach ($Property in @($Config.PSObject.Properties | Sort-Object Name)) {
        $Name = [string]$Property.Name
        if ($Skip -contains $Name -or $null -eq $Property.Value) {
            continue
        }
        $Option = "--" + $Name.Replace('_', '-')
        $Value = $Property.Value
        if ($Value -is [bool]) {
            if ($Value) {
                $Arguments.Add($Option)
            } elseif ($BooleanOptional -contains $Name) {
                $Arguments.Add("--no-" + $Name.Replace('_', '-'))
            }
            continue
        }
        if ($Value -is [System.Array]) {
            foreach ($Item in $Value) {
                $Arguments.Add($Option)
                $Arguments.Add([string]$Item)
            }
            continue
        }
        $Arguments.Add($Option)
        $Arguments.Add([string]$Value)
    }
    $Arguments.Add("--epochs")
    $Arguments.Add([string]$RequestedEpochs)
    $Arguments.Add("--device")
    $Arguments.Add($Device)
    $Arguments.Add("--output")
    $Arguments.Add($CheckpointPath)
    $Arguments.Add("--resume")
    $Arguments.Add($ResumeCheckpoint)
    return @($Arguments)
}

function Get-FreshTrainingArguments {
    param([Parameter(Mandatory = $true)][int]$RequestedEpochs)
    $Arguments = @(
        "scripts/train_v16.py",
        "--epochs", [string]$RequestedEpochs,
        "--proposal-epochs", [string]$FreshProposalEpochs,
        "--train-size", "1500",
        "--val-size", "500",
        "--batch-size", [string]$FreshBatchSize,
        "--num-points", "192",
        "--min-control-points", "8",
        "--max-control-points", "28",
        "--candidate-knots", "64",
        "--mse-tolerance", "5e-5",
        "--knot-match-tolerance", "0.01",
        "--certified-minimal-source",
        "--minimality-margin", "0.2",
        "--minimality-max-attempts", "16",
        "--minimality-audit-points", "512",
        "--oscillation-amplitude", "0.3",
        "--proposal-pass-target", "0.90",
        "--deployment-pass-target", "0.90",
        "--one-shot-selection-policy", "mass_topk",
        "--one-shot-safety-sigma", "0.20",
        "--one-shot-safety-knots", "2",
        "--final-safety-sigma", "0.05",
        "--final-safety-knots", "0",
        "--safety-anneal-epochs", "8",
        "--complexity-ramp-epochs", "8",
        "--complexity-max-scale", "4.0",
        "--complexity-pass-margin", "0.02",
        "--one-shot-coverage-bins", "4",
        "--min-selected-knots", "4",
        "--policy-samples", "2",
        "--counterfactual-edits", "2",
        "--teacher-prefix-search-steps", "6",
        "--real-fraction", "0",
        "--no-resample-train-each-epoch",
        "--torch-num-threads", "4",
        "--device", $Device,
        "--output", $CheckpointPath
    )
    $CandidateInit = $InitCheckpoint
    if (-not [string]::IsNullOrWhiteSpace($CandidateInit)) {
        if (-not [System.IO.Path]::IsPathRooted($CandidateInit)) {
            $CandidateInit = Join-Path $RepositoryRoot $CandidateInit
        }
        if (Test-Path -LiteralPath $CandidateInit) {
            $Arguments += @("--init-checkpoint", [System.IO.Path]::GetFullPath($CandidateInit))
        } else {
            Write-Warning "Optional proposal initializer was not found: $CandidateInit. Training will start from scratch."
        }
    }
    return $Arguments
}

function Test-FormalCheckpoint {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Phase = (
        "inspect_" + [System.IO.Path]::GetFileNameWithoutExtension($Path) + "_" +
        [Guid]::NewGuid().ToString("N").Substring(0, 8)
    )
    $Qualified = Invoke-LoggedPython -Phase $Phase `
        -CommandArguments @(
            "scripts/inspect_v16_checkpoint.py",
            "--checkpoint", $Path,
            "--required-pass-rate", "0.90",
            "--mse-tolerance", "5e-5"
        ) -AllowFailure
    if ($Qualified) {
        return $true
    }
    $PhaseLog = Join-Path $LogDirectory ($Phase + ".log")
    $AuditOutput = $(
        if (Test-Path -LiteralPath $PhaseLog) {
            Get-Content -LiteralPath $PhaseLog -Raw -Encoding utf8
        } else {
            ""
        }
    )
    if ($AuditOutput -match 'eligible:\s+NO') {
        return $false
    }
    throw "Checkpoint audit failed to load or validate $Path; this is not a normal unqualified-checkpoint result."
}

function Find-QualifiedCheckpoint {
    foreach ($Candidate in @($CheckpointPath, $LastPath, $ProposalPath)) {
        if ((Test-Path -LiteralPath $Candidate) -and (Test-FormalCheckpoint -Path $Candidate)) {
            return $Candidate
        }
    }
    return $null
}

$TranscriptStarted = $false
$PipelineStarted = Get-Date
try {
    if (-not $DryRun) {
        try {
            $LockStream = [System.IO.File]::Open(
                $LockPath,
                [System.IO.FileMode]::OpenOrCreate,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
            $LockStream.SetLength(0)
            $LockBytes = [Text.Encoding]::UTF8.GetBytes("PID=$PID`n")
            $LockStream.Write($LockBytes, 0, $LockBytes.Length)
            $LockStream.Flush()
        } catch {
            throw "Another overnight orchestrator already owns $LockPath; no new job was started."
        }
        if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
            throw "Python command was not found: $Python"
        }
        try {
            Start-Transcript -LiteralPath $MasterLog -Append | Out-Null
            $TranscriptStarted = $true
        } catch {
            Write-Warning "PowerShell transcript could not start; per-phase logs will still be written."
        }
    }
    Save-RunState
    Write-Host "v16 overnight profile: Kc=64 internal candidates; source K=4..24; MSE=5e-5."
    Write-Host "Fresh run uses train=1500, val=500, batch=$FreshBatchSize, target/fallback epochs=$TargetEpochs/$FallbackEpochs."
    Write-Host "Measured GTX 1070 budget: approximately 4.4-5.1 h training + 3.5-5.5 h paired benchmark; an RTX 3090 is normally faster."

    Wait-ForExistingTargetTraining

    $TrainingSucceeded = $true
    if (-not $SkipTraining) {
        $CompletedEpoch = Get-CompletedEpoch
        if ($CompletedEpoch -ge $TargetEpochs -and (Test-Path -LiteralPath $CheckpointPath)) {
            Write-Host "Training already reached epoch $CompletedEpoch; skipping directly to evaluation."
            $RunState.phases.training = [ordered]@{
                status = "already_complete"
                completed_epoch = $CompletedEpoch
            }
            Save-RunState
        } elseif (Test-Path -LiteralPath $LastPath) {
            Write-Host "Resuming epoch $($CompletedEpoch + 1)..$TargetEpochs from $LastPath."
            # Resume must replay the saved configuration exactly. In particular,
            # an existing batch-32 run is never silently changed to batch 64.
            $ResumeArguments = @("scripts/train_v16.py") + (Get-ResumeArguments `
                -ResumeCheckpoint $LastPath -RequestedEpochs $TargetEpochs)
            Assert-NoForeignGpuWork
            $TrainingSucceeded = Invoke-LoggedPython -Phase "train_resume" `
                -CommandArguments $ResumeArguments -AllowFailure
        } elseif (
            (Test-Path -LiteralPath $CheckpointPath) -or
            (Test-Path -LiteralPath $ProposalPath) -or
            (Test-Path -LiteralPath $HistoryPath)
        ) {
            Write-Warning (
                "Training artifacts exist but no resumable .last.pt is available. " +
                "They will not be deleted; evaluation will use the best available diagnostic checkpoint."
            )
            $TrainingSucceeded = $false
        } else {
            Assert-NoForeignGpuWork
            $TrainingSucceeded = Invoke-LoggedPython -Phase "train_fresh" `
                -CommandArguments (Get-FreshTrainingArguments -RequestedEpochs $TargetEpochs) -AllowFailure
        }
    } else {
        Write-Host "Training skipped by -SkipTraining."
        $RunState.phases.training = [ordered]@{ status = "skipped" }
        Save-RunState
    }

    $ExistingCandidates = @(
        @($CheckpointPath, $LastPath, $ProposalPath) |
            Where-Object { Test-Path -LiteralPath $_ }
    )
    if ($DryRun -and $ExistingCandidates.Count -eq 0) {
        $ExistingCandidates = @($CheckpointPath)
    }
    if ($ExistingCandidates.Count -eq 0) {
        throw "No v16 checkpoint exists after training; benchmark and figures cannot run."
    }

    if ($DryRun) {
        $SelectedCheckpoint = $CheckpointPath
        $Diagnostic = $false
    } else {
        $SelectedCheckpoint = Find-QualifiedCheckpoint
        $Diagnostic = $null -eq $SelectedCheckpoint
    }

    # A short second curriculum is attempted only if the formal audit failed
    # and at least 6.5 hours remain.  The reserve protects the full-strength
    # 66-curve published-method benchmark instead of silently weakening it.
    $ElapsedHours = ((Get-Date) - $PipelineStarted).TotalHours
    $HoursRemaining = $BudgetHours - $ElapsedHours
    $CompletedEpoch = Get-CompletedEpoch
    if (
        $Diagnostic -and
        -not $SkipTraining -and
        $FallbackEpochs -gt $CompletedEpoch -and
        (Test-Path -LiteralPath $LastPath) -and
        $HoursRemaining -ge 6.5
    ) {
        Write-Host (
            "Target-epoch checkpoint is not formally qualified. " +
            "Budget has $([math]::Round($HoursRemaining, 2)) h remaining; " +
            "continuing to fallback epoch $FallbackEpochs."
        )
        Assert-NoForeignGpuWork
        $FallbackArguments = @("scripts/train_v16.py") + (Get-ResumeArguments `
            -ResumeCheckpoint $LastPath -RequestedEpochs $FallbackEpochs)
        $FallbackSucceeded = Invoke-LoggedPython -Phase "train_fallback" `
            -CommandArguments $FallbackArguments -AllowFailure
        $RunState["fallback_training_succeeded"] = $FallbackSucceeded
        Save-RunState
        $SelectedCheckpoint = Find-QualifiedCheckpoint
        $Diagnostic = $null -eq $SelectedCheckpoint
    } elseif ($Diagnostic -and $FallbackEpochs -gt $CompletedEpoch) {
        Write-Warning (
            "The formal audit failed, but the fallback curriculum was skipped to protect " +
            "the benchmark reserve (remaining=$([math]::Round($HoursRemaining, 2)) h; required=6.5 h)."
        )
    }

    if ($null -eq $SelectedCheckpoint) {
        foreach ($Candidate in @($CheckpointPath, $LastPath, $ProposalPath)) {
            if (Test-Path -LiteralPath $Candidate) {
                $SelectedCheckpoint = $Candidate
                break
            }
        }
        Write-Warning (
            "No checkpoint passed the formal v16 audit. Continuing with $SelectedCheckpoint; " +
            "all reports and PNGs will be visibly marked DIAGNOSTIC NOT FINAL."
        )
    }

    $Hash = $(
        if ($DryRun -and -not (Test-Path -LiteralPath $SelectedCheckpoint)) {
            "0000000000000000000000000000000000000000000000000000000000000000"
        } else {
            (Get-FileHash -LiteralPath $SelectedCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    )
    $Tag = $(if ($Diagnostic) { "diagnostic_" + $Hash.Substring(0, 12) } else { "formal_" + $Hash.Substring(0, 12) })
    $ComparisonDirectory = Join-Path $ComparisonRoot $Tag
    $MethodFigureDirectory = Join-Path (Join-Path $FigureRoot $Tag) "method_comparison"
    $CaseFigureDirectory = Join-Path (Join-Path $FigureRoot $Tag) "ours_cases"
    foreach ($directory in @($ComparisonDirectory, $MethodFigureDirectory, $CaseFigureDirectory)) {
        [void](New-Item -ItemType Directory -Path $directory -Force)
    }
    $RunState.checkpoint = $SelectedCheckpoint
    $RunState.diagnostic_not_final = $Diagnostic
    $RunState.comparison_directory = $ComparisonDirectory
    $RunState.figure_directory = Join-Path $FigureRoot $Tag
    $RunState["training_command_succeeded"] = $TrainingSucceeded
    Save-RunState

    $DiagnosticArguments = @()
    if ($Diagnostic) {
        $DiagnosticArguments = @("--allow-unqualified-diagnostic")
    }
    $BenchmarkArguments = @(
        "scripts/benchmark_v16_datasets.py",
        "--checkpoint", $SelectedCheckpoint,
        "--output-dir", $ComparisonDirectory,
        "--samples-per-knot-count", [string]$SyntheticSamplesPerK,
        "--min-knot-count", "4",
        "--max-knot-count", "24",
        "--real-samples-per-dataset", [string]$RealSamplesPerDataset,
        "--mse-tolerance", "5e-5",
        "--max-internal-knots", "64",
        "--gradient-steps", "12",
        "--paper-initial-knots", "64",
        "--paper-admm-iterations", [string]$KangAdmmIterations,
        "--paper-lambda-bisections", "8",
        "--paper-relocation-iterations", "8",
        "--liang-dense-knots", "64",
        "--liang-feature-samples", "1025",
        "--dung-scan-intervals", "10",
        "--dung-optimization-iterations", "10",
        "--luo-de-population", "10",
        "--luo-de-iterations", [string]$LuoDeIterations,
        "--network-warmups", "3",
        "--network-repeats", "20",
        "--end-to-end-repeats", "1",
        "--torch-num-threads", "1",
        "--device", $Device,
        "--resume"
    ) + $DiagnosticArguments
    Assert-NoForeignGpuWork
    [void](Invoke-LoggedPython -Phase "benchmark_all_methods" -CommandArguments $BenchmarkArguments)

    $ComparisonReport = Join-Path $ComparisonDirectory "comparison.json"
    $PlotArguments = @(
        "scripts/plot_v16_method_comparison.py",
        "--input", $ComparisonReport,
        "--output-dir", $MethodFigureDirectory,
        "--method-set", "all",
        "--dpi", "300",
        "--reference"
    ) + $DiagnosticArguments
    [void](Invoke-LoggedPython -Phase "plot_method_comparison" -CommandArguments $PlotArguments)

    $CaseArguments = @(
        "scripts/visualize_v16_ours_cases.py",
        "--checkpoint", $SelectedCheckpoint,
        "--output-dir", $CaseFigureDirectory,
        "--real-samples-per-dataset", "2",
        "--selection-seed", "20260910",
        "--mse-tolerance", "5e-5",
        "--max-internal-knots", "64",
        "--network-warmups", "2",
        "--network-repeats", "5",
        "--end-to-end-repeats", "1",
        "--torch-num-threads", "1",
        "--device", $Device,
        "--dpi", "300",
        "--overwrite"
    ) + $DiagnosticArguments
    [void](Invoke-LoggedPython -Phase "plot_ours_cases" -CommandArguments $CaseArguments)

    $RunState.status = "completed"
    $RunState["finished_utc"] = [DateTime]::UtcNow.ToString("o")
    Save-RunState
    Write-Host ""
    Write-Host "Overnight pipeline completed."
    Write-Host "  checkpoint: $SelectedCheckpoint"
    Write-Host "  diagnostic: $Diagnostic"
    Write-Host "  comparison: $ComparisonDirectory"
    Write-Host "  figures:    $(Join-Path $FigureRoot $Tag)"
    Write-Host "  manifest:   $ManifestPath"
} catch {
    $RunState.status = "failed"
    $RunState["finished_utc"] = [DateTime]::UtcNow.ToString("o")
    $RunState["error"] = $_.Exception.Message
    Save-RunState
    Write-Error $_
    exit 1
} finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
    if ($null -ne $LockStream) {
        $LockStream.Dispose()
    }
}
