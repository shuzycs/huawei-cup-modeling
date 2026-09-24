# 问题 2 实验的端到端运行脚本。
#
# 用法（在仓库根目录）：
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage all
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage env       # 只写环境清单
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage data      # 数据核查 + 准备
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage smoke     # 冒烟测试
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage train     # E0..E3 x seed 42/52/62
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage eval      # 固定验证视图评估
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage analyze   # 指标表与图
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage test      # 附件 2 test 独立检验
#   powershell -ExecutionPolicy Bypass -File scripts/run_problem2.ps1 -Stage infer     # 附件 3 推理
#
# 每个实验/种子写入独立目录 outputs/problem2/runs/<E>_seed<seed>/，不覆盖上一项。
# 日志追加写入 outputs/problem2/logs/run_problem2.log。

param(
    [ValidateSet("all", "env", "data", "smoke", "train", "eval", "analyze", "test", "infer")]
    [string]$Stage = "all",
    [string]$Python = "",
    [int[]]$Seeds = @(42, 52, 62),
    [string[]]$Experiments = @("E0", "E1", "E2", "E3", "E4", "E5"),
    [string]$Config = "configs/problem2.yaml",
    [int]$MaxEpochs = 0,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($Python)) {
    $candidates = @(
        "$env:USERPROFILE\anaconda3\envs\shuzy-hcm\python.exe",
        "D:\WorkSoftware\anaconda\envs\shuzy-hcm\python.exe",
        "$env:CONDA_PREFIX\python.exe"
    )
    $Python = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $Python) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source }
    }
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "找不到 shuzy-hcm 环境的 python.exe，请用 -Python 指定完整路径"
}

$env:PYTHONIOENCODING = "utf-8"
$LogDir = Join-Path $Root "outputs\problem2\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "run_problem2.log"

function Invoke-Py {
    param([string[]]$PyArgs, [string]$Label)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label :: $Python $($PyArgs -join ' ')"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python @PyArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    $code = $LASTEXITCODE
    $sw.Stop()
    $done = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label exit=$code elapsed=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
    Write-Host $done
    Add-Content -Path $LogFile -Value $done -Encoding UTF8
    if ($code -ne 0) { throw "$Label 失败（exit=$code）" }
}

Write-Host "Python : $Python"
# 环境自检。注意：不要在 PowerShell 双引号字符串里写 Python 三元表达式
# （`a if cond else b`），pwsh 解析器会报 Unexpected token；这里改成两条独立语句。
& $Python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
& $Python -c "import torch; print('device', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('device cpu')"
if ($LASTEXITCODE -ne 0) { throw "无法导入 torch，请检查 -Python 是否指向 shuzy-hcm 环境（当前：$Python）" }

if ($Stage -in @("all", "env")) {
    Invoke-Py @("-m", "src.problem2.report", "--config", $Config) "环境清单"
}

if ($Stage -in @("all", "data")) {
    Invoke-Py @("-m", "src.problem2.inspect_data", "--config", $Config) "数据核查"
    Invoke-Py @("-m", "src.problem2.prepare_data", "--config", $Config) "数据准备"
}

if ($Stage -in @("all", "smoke")) {
    $smokeArgs = @("-m", "src.problem2.train", "--config", $Config, "--experiment", "E3", "--seed", "42", "--smoke", "--max-epochs", "1")
    Invoke-Py $smokeArgs "冒烟测试"
}

if ($Stage -in @("all", "train")) {
    # E5 使用经验证集小范围搜索选定的覆盖项（记录在各自 run 目录的 run_info.json 的 overrides 字段）
    $e5Overrides = @("--set", "text_model.unfrozen_learning_rate=0.001",
                     "--set", "training.class_weighted_loss=true",
                     "--set", "text_model.min_lr_ratio=0.2")
    foreach ($exp in $Experiments) {
        foreach ($seed in $Seeds) {
            $runDir = Join-Path $Root "outputs\problem2\runs\${exp}_seed${seed}"
            if ($SkipExisting -and (Test-Path (Join-Path $runDir "best.safetensors"))) {
                Write-Host "跳过已存在: $runDir"
                continue
            }
            $args = @("-m", "src.problem2.train", "--config", $Config, "--experiment", $exp, "--seed", "$seed")
            if ($MaxEpochs -gt 0) { $args += @("--max-epochs", "$MaxEpochs") }
            if ($exp -eq "E5") { $args += $e5Overrides }
            if ($exp -eq "E4") {
                # 教师必须先在相同 seed 上训练好（幂等：已训练则跳过）
                $teacherCkpt = Join-Path $Root "outputs\problem2\teacher\mosei_text768_seed${seed}\best.safetensors"
                if (-not (Test-Path $teacherCkpt)) {
                    Invoke-Py @("-m", "src.problem2.train", "--config", $Config,
                        "--experiment", "E4", "--seed", "$seed", "--train-teacher") "训练教师 seed=$seed"
                }
                $args += "--teacher"
            }
            Invoke-Py $args "训练 $exp seed=$seed"
        }
    }
}

if ($Stage -in @("all", "eval")) {
    foreach ($exp in $Experiments) {
        foreach ($seed in $Seeds) {
            Invoke-Py @("-m", "src.problem2.evaluate", "--config", $Config, "--split", "valid",
                "--experiment", $exp, "--seed", "$seed") "评估 $exp seed=$seed"
        }
    }
}

if ($Stage -in @("all", "analyze")) {
    Invoke-Py @("-m", "src.problem2.analyze", "--config", $Config) "规律分析与图表"
}

if ($Stage -in @("all", "test")) {
    foreach ($exp in $Experiments) {
        Invoke-Py @("-m", "src.problem2.evaluate", "--config", $Config, "--split", "test",
            "--experiment", $exp, "--seed", "$($Seeds[0])") "附件2 test 检验 $exp"
    }
}

if ($Stage -in @("all", "infer")) {
    $ckptExp = if ($Experiments -contains "E3") { "E3" } else { $Experiments[-1] }
    Invoke-Py @("-m", "src.problem2.infer", "--config", $Config,
        "--experiment", $ckptExp,
        "--checkpoint", "outputs/problem2/runs/${ckptExp}_seed$($Seeds[0])\best.safetensors") "附件 3 推理"
}

Write-Host "阶段 '$Stage' 完成。日志：$LogFile"
