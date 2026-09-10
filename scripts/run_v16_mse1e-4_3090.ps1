# Run a fresh v16 MSE=1e-4 experiment and its six-method evaluation on Windows.
#
# This orchestrator is intentionally non-destructive: it refuses to start when
# any run-owned checkpoint, log, comparison, or figure artifact already exists.
# It never resumes and never passes an overwrite flag.  A compatible proposal
# checkpoint may initialize only the encoder, ParameterHead and CandidateHead;
# its ordered interval-query table is rank-interpolated when Kc differs, while
# deterministic target anchors, the selector and optimizer always start fresh.
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$RunName = "candidate_selection_v16_mse1e-4_k56_ordered_highk_softcost",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [int]$Epochs = 104,
    [int]$ProposalEpochs = 40,
    [int]$TrainSize = 3000,
    [int]$ValSize = 600,
    [int]$RealValSize = 100,
    [double]$RealFraction = 0.35,
    [double]$ProposalHighKFraction = 0.50,
    [int]$ProposalHighKMinKnots = 40,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 4,
    [int]$SyntheticSamplesPerK = 5,
    [int]$RealSamplesPerDataset = 20,
    [int]$VisualSamplesPerDataset = 2,
    [int]$KangAdmmIterations = 1000,
    [int]$LuoDePopulation = 20,
    [int]$LuoDeIterations = 100,
    [int]$NetworkRepeats = 100,
    [int]$EndToEndRepeats = 3,
    [string]$OutputRoot = "",
    [string]$InitCheckpoint = (
        "outputs/checkpoints/" +
        "candidate_selection_v16_mse5e-5_k64.proposal.pt"
    ),
    [switch]$Diagnostic,
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
foreach ($Entry in @{
    Epochs = $Epochs
    ProposalEpochs = $ProposalEpochs
    TrainSize = $TrainSize
    ValSize = $ValSize
    RealValSize = $RealValSize
    ProposalHighKMinKnots = $ProposalHighKMinKnots
    BatchSize = $BatchSize
    SyntheticSamplesPerK = $SyntheticSamplesPerK
    RealSamplesPerDataset = $RealSamplesPerDataset
    VisualSamplesPerDataset = $VisualSamplesPerDataset
    KangAdmmIterations = $KangAdmmIterations
    LuoDePopulation = $LuoDePopulation
    LuoDeIterations = $LuoDeIterations
    NetworkRepeats = $NetworkRepeats
    EndToEndRepeats = $EndToEndRepeats
}.GetEnumerator()) {
    if ($Entry.Value -lt 1) {
        throw "$($Entry.Key) must be positive."
    }
}
if ($ProposalEpochs -ge $Epochs) {
    throw "ProposalEpochs must be smaller than Epochs."
}
if ($NumWorkers -lt 0) {
    throw "NumWorkers must be non-negative."
}
if ($LuoDePopulation -lt 5) {
    throw "LuoDePopulation must be at least 5."
}
if (
    [double]::IsNaN($RealFraction) -or
    [double]::IsInfinity($RealFraction) -or
    $RealFraction -lt 0.0 -or
    $RealFraction -gt 1.0
) {
    throw "RealFraction must lie in [0,1]."
}
if (
    [double]::IsNaN($ProposalHighKFraction) -or
    [double]::IsInfinity($ProposalHighKFraction) -or
    $ProposalHighKFraction -lt 0.0 -or
    $ProposalHighKFraction -gt 1.0
) {
    throw "ProposalHighKFraction must lie in [0,1]."
}
if ($ProposalHighKMinKnots -lt 4 -or $ProposalHighKMinKnots -gt 56) {
    throw "ProposalHighKMinKnots must lie inside source K=4..56."
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
$CheckpointPath = Join-Path $CheckpointDirectory ($RunName + ".pt")
$CheckpointFamily = @(
    $CheckpointPath,
    (Join-Path $CheckpointDirectory ($RunName + ".last.pt")),
    (Join-Path $CheckpointDirectory ($RunName + ".proposal.pt")),
    (Join-Path $CheckpointDirectory ($RunName + ".history.json"))
)

# Fail before creating anything if this run name already owns artifacts.  This
# prevents an interrupted or completed experiment from being silently replaced.
$OwnedRoots = @($LogDirectory, $ComparisonRoot, $FigureRoot)
$Collisions = @(
    @($CheckpointFamily + $OwnedRoots) |
        Where-Object { Test-Path -LiteralPath $_ }
)
if ($Collisions.Count -gt 0) {
    throw (
        "Refusing to overwrite existing run artifacts. Choose a new -RunName or " +
        "-OutputRoot:`n  " + ($Collisions -join "`n  ")
    )
}

foreach ($Directory in @($CheckpointDirectory, $LogDirectory)) {
    [void](New-Item -ItemType Directory -Path $Directory -Force)
}
$ManifestPath = Join-Path $LogDirectory "pipeline_manifest.json"

$UjiManifest = Join-Path $RepositoryRoot "data/splits/uji_pen_v2.jsonl"
$NaturalEarthManifest = Join-Path $RepositoryRoot (
    "data/processed/natural_earth/v5.1.2_10m_coastline/manifest.jsonl"
)
$UsgsManifest = Join-Path $RepositoryRoot (
    "data/processed/usgs_contours/large_scale/manifest.jsonl"
)
foreach ($Manifest in @($UjiManifest, $NaturalEarthManifest, $UsgsManifest)) {
    if (-not $DryRun -and -not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "Required real-data manifest does not exist: $Manifest"
    }
}

$RunState = [ordered]@{
    run_name = $RunName
    started_utc = [DateTime]::UtcNow.ToString("o")
    status = "initializing"
    diagnostic_requested = [bool]$Diagnostic
    checkpoint = [System.IO.Path]::GetFullPath($CheckpointPath)
    requested_profile = [ordered]@{
        simplification_contract = "ranked_prefix_ordered_proposal_high_k_soft_subset_cost_v3"
        checkpoint_selection = "mean_per_curve_subset_cost_v1"
        qualification_contract = "v16_structural_integrity_pass_rates_report_only_v3"
        aggregate_pass_role = "reporting_reference_only"
        simplification_curriculum = "deterministic_linear_by_joint_epoch"
        aggregate_pass_feedback = $false
        mse_tolerance = 1e-4
        candidate_internal_knots = 56
        full_cubic_knot_vector_size_at_all_keep = 64
        source_internal_knots = "4..56"
        source_control_points = "8..60"
        knot_min_span = 0.01
        points = 192
        epochs = $Epochs
        proposal_epochs = $ProposalEpochs
        train_size = $TrainSize
        validation_size = $ValSize
        synthetic_boundary_validation_size = [Math]::Min(32, $ValSize)
        real_validation_size_per_source = $RealValSize
        real_fraction = $RealFraction
        proposal_high_k_fraction = $ProposalHighKFraction
        proposal_high_k_min_knots = $ProposalHighKMinKnots
        proposal_knot_assignment_weight = 1.0
        batch_size = $BatchSize
        selection_policy = "mass_topk"
        initial_keep_fraction = 0.5357142857142857
        teacher_low_count_sweep = 16
        synthetic_count_role = "upper_bound"
        synthetic_geometry_oracle_teacher = $true
        oracle_teacher_extra_knots = 2
        coverage_bins = 0
        synthetic_samples_per_k = $SyntheticSamplesPerK
        real_samples_per_dataset = $RealSamplesPerDataset
        device = $Device
    }
    phases = [ordered]@{}
}

function Save-RunState {
    $RunState | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $ManifestPath -Encoding utf8
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
        Tee-Object -FilePath $PhaseLog |
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

try {
    Save-RunState
    Write-Host (
        "Fresh v16 RTX 3090 profile: MSE=1e-4, Kc=56 internal " +
        "(64-entry full cubic knot vector), source K=4..56, " +
        "train/val=$TrainSize/$ValSize, batch=$BatchSize."
    )
    Write-Host (
        "The selector uses adaptive mass-TopK; simple-curve teachers exhaustively " +
        "check K<=16 and may use fewer knots than the certified source."
    )
    Write-Host (
        "Proposal synthetic sampling uses " +
        ([string]::Format(
            [Globalization.CultureInfo]::InvariantCulture,
            "{0:P0}",
            $ProposalHighKFraction
        )) +
        " from K>=$ProposalHighKMinKnots and monotone one-to-one knot " +
        "assignment; Joint restores the original K=4..56 distribution."
    )

    $TrainingArguments = @(
        "scripts/train_v16.py",
        "--epochs", [string]$Epochs,
        "--proposal-epochs", [string]$ProposalEpochs,
        "--train-size", [string]$TrainSize,
        "--val-size", [string]$ValSize,
        "--synthetic-boundary-val-size", "32",
        "--real-val-size", [string]$RealValSize,
        "--proposal-high-k-fraction", ([string]::Format(
            [Globalization.CultureInfo]::InvariantCulture, "{0:R}", $ProposalHighKFraction
        )),
        "--proposal-high-k-min-knots", [string]$ProposalHighKMinKnots,
        "--batch-size", [string]$BatchSize,
        "--num-points", "192",
        "--min-control-points", "8",
        "--max-control-points", "60",
        "--candidate-knots", "56",
        "--knot-min-span", "0.01",
        "--mse-tolerance", "1e-4",
        "--knot-match-tolerance", "0.01",
        "--certified-minimal-source",
        "--minimality-margin", "0.2",
        "--minimality-max-attempts", "16",
        "--minimality-audit-points", "512",
        "--oscillation-amplitude", "0.3",
        "--proposal-knot-assignment-weight", "1.0",
        "--one-shot-selection-policy", "mass_topk",
        "--initial-keep-fraction", "0.5357142857142857",
        "--teacher-low-count-sweep", "16",
        "--synthetic-count-role", "upper_bound",
        "--synthetic-geometry-oracle-teacher",
        "--oracle-teacher-extra-knots", "2",
        "--one-shot-coverage-bins", "0",
        "--min-selected-knots", "4",
        "--one-shot-safety-sigma", "0.20",
        "--one-shot-safety-knots", "2",
        "--final-safety-sigma", "0.03",
        "--final-safety-knots", "0",
        "--safety-anneal-epochs", "12",
        "--complexity-ramp-epochs", "12",
        "--complexity-max-scale", "6.0",
        "--policy-samples", "2",
        "--counterfactual-edits", "4",
        "--teacher-prefix-search-steps", "7",
        "--real-fraction", ([string]::Format(
            [Globalization.CultureInfo]::InvariantCulture, "{0:R}", $RealFraction
        )),
        "--real-manifest", $UjiManifest,
        "--real-manifest", $NaturalEarthManifest,
        "--real-manifest", $UsgsManifest,
        "--resample-train-each-epoch",
        "--num-workers", [string]$NumWorkers,
        "--torch-num-threads", "4",
        "--device", $Device,
        "--output", [System.IO.Path]::GetFullPath($CheckpointPath)
    )
    $ResolvedInit = $null
    if (-not [string]::IsNullOrWhiteSpace($InitCheckpoint)) {
        $ResolvedInit = $(
            if ([System.IO.Path]::IsPathRooted($InitCheckpoint)) {
                [System.IO.Path]::GetFullPath($InitCheckpoint)
            } else {
                [System.IO.Path]::GetFullPath(
                    (Join-Path $RepositoryRoot $InitCheckpoint)
                )
            }
        )
        if (Test-Path -LiteralPath $ResolvedInit -PathType Leaf) {
            Write-Host (
                "Warm-starting encoder, ParameterHead and CandidateHead only " +
                "(ordered query ranks adapt to Kc=56) from: " +
                $ResolvedInit
            )
            $TrainingArguments += @("--init-checkpoint", $ResolvedInit)
            $RunState["initializer"] = $ResolvedInit
        } else {
            Write-Warning (
                "Optional proposal initializer was not found: $ResolvedInit. " +
                "The fresh experiment will train all modules from scratch."
            )
            $RunState["initializer"] = $null
        }
    }
    [void](Invoke-LoggedPython -Phase "train_fresh" -CommandArguments $TrainingArguments)

    if (-not $DryRun -and -not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
        throw "Training completed without the requested best checkpoint: $CheckpointPath"
    }

    $InspectSucceeded = Invoke-LoggedPython -Phase "inspect_checkpoint" `
        -CommandArguments @(
            "scripts/inspect_v16_checkpoint.py",
            "--checkpoint", [System.IO.Path]::GetFullPath($CheckpointPath),
            "--mse-tolerance", "1e-4"
        ) -AllowFailure
    $RunState["checkpoint_integrity_passed"] = [bool]$InspectSucceeded
    Save-RunState
    if (-not $InspectSucceeded -and -not $Diagnostic) {
        throw (
            "The checkpoint failed the v16 structural-integrity contract, so no " +
            "benchmark or figure was produced. Inspection checks the Joint stage, " +
            "deterministic curriculum maturity, final safety, data/capacity contract, " +
            "selection rule and MSE tolerance. Pass rate, retained K and knot/count " +
            "accuracy remain reported diagnostics and cannot cause this failure. See " +
            "inspect_checkpoint.log. Use --allow-unqualified-diagnostic only in " +
            "separate troubleshooting commands."
        )
    }

    $Hash = $(
        if ($DryRun) {
            "0000000000000000000000000000000000000000000000000000000000000000"
        } else {
            (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    )
    $DiagnosticOutput = [bool]$Diagnostic -or -not $InspectSucceeded
    $Tag = $(
        if ($DiagnosticOutput) {
            "diagnostic_" + $Hash.Substring(0, 12)
        } else {
            "formal_" + $Hash.Substring(0, 12)
        }
    )
    $ComparisonDirectory = Join-Path $ComparisonRoot $Tag
    $MetricFigureDirectory = Join-Path (Join-Path $FigureRoot $Tag) "four_metrics"
    $RealFigureDirectory = Join-Path (Join-Path $FigureRoot $Tag) "six_method_real_cases"
    foreach ($Directory in @(
        $ComparisonDirectory, $MetricFigureDirectory, $RealFigureDirectory
    )) {
        if (Test-Path -LiteralPath $Directory) {
            throw "Refusing to overwrite evaluation output: $Directory"
        }
        [void](New-Item -ItemType Directory -Path $Directory -Force)
    }

    $DiagnosticArguments = @()
    if ($DiagnosticOutput) {
        $DiagnosticArguments = @("--allow-unqualified-diagnostic")
    }

    $CommonBaselineArguments = @(
        "--max-internal-knots", "56",
        "--gradient-steps", "12",
        "--paper-initial-knots", "56",
        "--paper-admm-iterations", [string]$KangAdmmIterations,
        "--paper-lambda-bisections", "10",
        "--paper-relocation-iterations", "12",
        "--liang-dense-knots", "56",
        "--liang-feature-samples", "1025",
        "--dung-scan-intervals", "10",
        "--dung-optimization-iterations", "10",
        "--luo-eta", "0.5",
        "--luo-de-population", [string]$LuoDePopulation,
        "--luo-de-iterations", [string]$LuoDeIterations
    )
    $ManifestArguments = @(
        "--manifest", ("UJI=" + $UjiManifest),
        "--manifest", ("NaturalEarth=" + $NaturalEarthManifest),
        "--manifest", ("USGS=" + $UsgsManifest)
    )

    $BenchmarkArguments = @(
        "scripts/benchmark_v16_datasets.py",
        "--checkpoint", [System.IO.Path]::GetFullPath($CheckpointPath),
        "--output-dir", $ComparisonDirectory,
        "--method-set", "published",
        "--samples-per-knot-count", [string]$SyntheticSamplesPerK,
        "--min-knot-count", "4",
        "--max-knot-count", "56",
        "--real-samples-per-dataset", [string]$RealSamplesPerDataset,
        "--mse-tolerance", "1e-4"
    ) + $ManifestArguments + $CommonBaselineArguments + @(
        "--network-warmups", "10",
        "--network-repeats", [string]$NetworkRepeats,
        "--end-to-end-repeats", [string]$EndToEndRepeats,
        "--torch-num-threads", "4",
        "--device", $Device
    ) + $DiagnosticArguments
    [void](Invoke-LoggedPython -Phase "benchmark_six_methods" `
        -CommandArguments $BenchmarkArguments)

    $ComparisonReport = Join-Path $ComparisonDirectory "comparison.json"
    $PlotArguments = @(
        "scripts/plot_v16_method_comparison.py",
        "--input", $ComparisonReport,
        "--output-dir", $MetricFigureDirectory,
        "--method-set", "published",
        "--dpi", "300",
        "--reference"
    ) + $DiagnosticArguments
    [void](Invoke-LoggedPython -Phase "plot_four_metrics" `
        -CommandArguments $PlotArguments)

    $VisualizationArguments = @(
        "scripts/visualize_v16_real_deployments.py",
        "--checkpoint", [System.IO.Path]::GetFullPath($CheckpointPath),
        "--output-dir", $RealFigureDirectory,
        "--real-samples-per-dataset", [string]$VisualSamplesPerDataset,
        "--selection-seed", "20260910",
        "--mse-tolerance", "1e-4"
    ) + $ManifestArguments + $CommonBaselineArguments + @(
        "--network-warmups", "10",
        "--network-repeats", [string]$NetworkRepeats,
        "--end-to-end-repeats", [string]$EndToEndRepeats,
        "--torch-num-threads", "4",
        "--device", $Device,
        "--dpi", "300"
    ) + $DiagnosticArguments
    [void](Invoke-LoggedPython -Phase "visualize_six_method_real_cases" `
        -CommandArguments $VisualizationArguments)

    $RunState.status = "completed"
    $RunState["diagnostic_not_final"] = $DiagnosticOutput
    $RunState["comparison_directory"] = $ComparisonDirectory
    $RunState["figure_directory"] = Join-Path $FigureRoot $Tag
    $RunState["finished_utc"] = [DateTime]::UtcNow.ToString("o")
    Save-RunState
    Write-Host ""
    Write-Host "v16 MSE=1e-4 pipeline completed."
    Write-Host "  checkpoint: $CheckpointPath"
    Write-Host "  diagnostic: $DiagnosticOutput"
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
}
