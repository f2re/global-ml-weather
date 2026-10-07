"""Strict numerical policy and exact repeated graph reductions (CUDA-first)."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils import deterministic
import global_weather.devices as devices

from global_weather.devices import (
    configure_training_numerics,
    select_device,
    training_runtime_identity,
)


@pytest.fixture(autouse=True)
def restore_numerical_settings():
    cublas_config = devices._CUBLAS_CONFIG_BEFORE_INIT
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = deterministic.fill_uninitialized_memory
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield
    devices._CUBLAS_CONFIG_BEFORE_INIT = cublas_config
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    deterministic.fill_uninitialized_memory = fill
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_training_policy_records_strict_actual_settings():
    configure_training_numerics("cpu")
    numerical = training_runtime_identity("cpu")["numerics"]
    assert numerical["policy"] == "strict-deterministic-1"
    assert numerical["deterministic_algorithms"] is True
    assert numerical["warn_only"] is False
    assert numerical["fill_uninitialized_memory"] is True
    assert numerical["cudnn_benchmark"] is False
    assert numerical["cudnn_deterministic"] is True
    assert numerical["float32_matmul_precision"] == "highest"
    assert numerical["matmul_allow_tf32"] is False
    assert numerical["cudnn_allow_tf32"] is False
    assert numerical["cublas_workspace_config"] is None


def test_cuda_workspace_config_set_before_initialization(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    configure_training_numerics("cuda:0")
    import os

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("config", [None, "", ":4096:2", "invalid"])
def test_cuda_workspace_rejects_unusable_initialized_context(monkeypatch, config):
    if config is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", config)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_training_numerics("cuda:0")


@pytest.mark.parametrize("config", [":4096:8", ":16:8"])
def test_cuda_workspace_preserves_operator_setting(monkeypatch, config):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", config)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    configure_training_numerics("cuda:0")
    import os

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == config


def test_cuda_workspace_rejects_changed_initialized_context(monkeypatch):
    config = ":16:8" if devices._CUBLAS_CONFIG_BEFORE_INIT != ":16:8" else ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", config)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="изменён после запуска CUDA"):
        configure_training_numerics("cuda:0")


def test_cuda_workspace_remembers_accepted_setting_before_initialization(monkeypatch):
    config = ":16:8" if devices._CUBLAS_CONFIG_BEFORE_INIT != ":16:8" else ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", config)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    configure_training_numerics("cuda:0")
    assert devices._CUBLAS_CONFIG_BEFORE_INIT == config
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    configure_training_numerics("cuda:0")
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


@pytest.mark.parametrize("mutation", ["missing_policy", "warn_only", "threads"])
def test_resume_rejects_changed_numerical_environment(tmp_path, monkeypatch, mutation):
    from global_weather.pipeline import runner

    configure_training_numerics("cpu")
    cfg = runner.TrainConfig(device="cpu")
    runtime = deepcopy(training_runtime_identity("cpu"))
    if mutation == "missing_policy":
        del runtime["numerics"]
    elif mutation == "warn_only":
        runtime["numerics"]["warn_only"] = True
    else:
        runtime["numerics"]["threads"] += 1
    state = {
        "schema": "weather-training-state-1",
        "dataset_fingerprint": "dataset",
        "config_fingerprint": runner.digest(cfg.identity()),
        "runtime": runtime,
        "software": {},
        "data_kind": "synthetic",
    }
    monkeypatch.setattr(runner, "read_json", lambda path: state)
    monkeypatch.setattr(runner, "artifact", lambda run, pointer: run / "state.json")
    monkeypatch.setattr(runner, "software", lambda: {})
    with pytest.raises(ValueError, match="численная среда"):
        runner._load_training_state(
            tmp_path, SimpleNamespace(fingerprint="dataset", kind="synthetic"), cfg
        )


def test_graph_reduction_forward_and_backward_are_exactly_repeatable():
    # The remote GPU must exercise CUDA. A host without CUDA uses the normal
    # auto policy; there is no skip or substitution on an available GPU.
    device = select_device("auto")
    configure_training_numerics(device)
    torch.manual_seed(173)
    source = torch.randn(2048, 8, device=device)
    indices = torch.arange(2048, device=device) % 13
    results = []
    for _ in range(4):
        x = source.clone().requires_grad_(True)
        reduced = x.new_zeros(13, 8).index_add(0, indices, x)
        # Repeated indexing also exercises the gather/index backward path.
        loss = reduced[indices].square().mean()
        loss.backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        results.append((reduced.detach().clone(), x.grad.clone()))
    for output, gradient in results[1:]:
        assert torch.equal(output, results[0][0])
        assert torch.equal(gradient, results[0][1])
