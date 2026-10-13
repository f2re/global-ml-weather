#!/bin/bash
set -u
cd "$1" || exit 1
mkdir -p outputs/verification-qa
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/home/user/global-weather-audit-20261012-venv/lib/python3.11/site-packages
p=/home/user/global-ml-weather/.venv/bin/python
q=outputs/verification-qa
date -u +%FT%TZ > "$q/started.txt"
run() { "$@"; }
$p -m pytest tests/test_joint_monthly_grib_check.py tests/test_joint_monthly_verification.py tests/test_joint_training_audit.py tests/test_joint_verification_recount.py tests/test_joint_monthly_report.py -q > "$q/focused.log" 2>&1 &&
$p -m pytest -q > "$q/full-pytest.log" 2>&1 &&
$p -m pytest tests/test_ecosystem.py tests/test_agent_contracts.py tests/test_pipeline.py tests/test_reference_assets.py -q > "$q/contracts.log" 2>&1 &&
$p -m global_weather.cli train-smoke --optimizer-steps 2 --horizon-hours 72 --output outputs/baseline-check > "$q/baseline.log" 2>&1 &&
$p -m global_weather.adaptive_cli --horizon-hours 72 --output outputs/adaptive-check > "$q/adaptive.log" 2>&1 &&
$p -m global_weather.multimodal selftest --output outputs/multimodal-check > "$q/multimodal.log" 2>&1 &&
$p scripts/pipeline-test.py > "$q/pipeline.log" 2>&1 &&
$p scripts/check_docs.py --strict > "$q/docs.log" 2>&1
status=$?
echo "$status" > "$q/matrix-exit.txt"
date -u +%FT%TZ > "$q/matrix-finished.txt"
if [ "$status" -ne 0 ]; then tail -40 "$q/focused.log"; tail -40 "$q/full-pytest.log"; exit "$status"; fi
CUDA_VISIBLE_DEVICES=0 $p -m pytest tests/test_profile_verification_review.py -q > "$q/cuda-check.log" 2>&1
status=$?
echo "$status" > "$q/cuda-exit.txt"
tail -3 "$q/full-pytest.log"
tail -3 "$q/cuda-check.log"
exit "$status"
