#!/usr/bin/env python3
"""Подготовка и калибровка контура ассимиляции спутниковых данных.

Инициализирует адаптер ArktikaElektroMSUGSAdapter, блок коррекции смещений VarBC,
операторы ИКФС-2 и МТВЗА-ГЯ, связывает контрольную точку mesh8 с ассимиляционным
функционалом SatelliteAssimilationLoss.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from global_weather.multimodal.ru_sat_msugs import ArktikaElektroMSUGSAdapter, ViewConditionedFiLMBlock
from global_weather.analysis.observation_operator import (
    VariationalBiasCorrection,
    IKFS2CloudClearedOperator,
    MeteorMTVZAAllSkyOperator,
)
from global_weather.losses import SatelliteAssimilationLoss


def build_satellite_assimilation_pipeline(
    hidden_dim: int = 16,
    num_selected_ikfs2: int = 180,
    num_mtvza: int = 29,
) -> dict[str, nn.Module]:
    """Создает полный ансамбль спутниковых операторов и функционалов потерь."""
    msugs_adapter = ArktikaElektroMSUGSAdapter()
    film_block = ViewConditionedFiLMBlock(hidden_dim=hidden_dim, geom_dim=4)
    ikfs2_operator = IKFS2CloudClearedOperator(num_selected_channels=num_selected_ikfs2, num_pcs=32)
    mtvza_operator = MeteorMTVZAAllSkyOperator(num_channels=num_mtvza)
    loss_fn = SatelliteAssimilationLoss(huber_delta=1.5, gross_error_sigma=3.5, varbc_reg_weight=1e-3)

    return {
        "msugs_adapter": msugs_adapter,
        "film_block": film_block,
        "ikfs2_operator": ikfs2_operator,
        "mtvza_operator": mtvza_operator,
        "loss_fn": loss_fn,
    }


def verify_assimilation_step(pipeline: dict[str, nn.Module]) -> dict:
    """Выполняет проверочный прогон ассимиляционного шага с обратным распространением."""
    msugs = pipeline["msugs_adapter"]
    film = pipeline["film_block"]
    ikfs2 = pipeline["ikfs2_operator"]
    mtvza = pipeline["mtvza_operator"]
    loss_fn = pipeline["loss_fn"]

    # 1. Проверка композитирования МСУ-ГС Электро-Л + Арктика-М
    b, n_sats, n_nodes = 2, 2, 42
    raw_channels = torch.full((b, n_sats, n_nodes, 10), 50.0, requires_grad=True)
    valid_mask = torch.ones((b, n_sats, n_nodes), dtype=torch.bool)
    node_xyz = torch.randn(n_nodes, 3)
    node_norm = node_xyz / torch.linalg.norm(node_xyz, dim=-1, keepdim=True)
    sat_xyz = torch.tensor([[[0.0, 0.0, 42164.0], [0.0, 30000.0, 38000.0]]] * b)
    sun_dir = torch.tensor([[0.0, 0.0, 1.0]] * b)
    slope = torch.ones(n_sats, 7)
    offset = torch.zeros(n_sats, 7)

    fused_features = msugs(raw_channels, valid_mask, node_xyz, node_norm, sat_xyz, sun_dir, slope, offset)
    assert fused_features.shape == (b, n_nodes, 20), f"Неверная размерность композита: {fused_features.shape}"

    # 2. Модуляция признаков через FiLM
    h_state = torch.randn(b, n_nodes, 16, requires_grad=True)
    geom_features = fused_features[..., 14:18]  # eff_mu_sat, eff_air_mass, eff_r_norm, mu_sun
    h_modulated = film(h_state, geom_features)

    # 3. Оператор МТВЗА-ГЯ All-Sky
    n_obs = 8
    obs_tb = torch.full((n_obs, 29), 260.0)
    model_profiles = torch.randn(n_obs, 37 * 3, requires_grad=True)
    surf_state = torch.tensor([[285.0, 6.0, 0.0, 0.0]] * n_obs, requires_grad=True)
    varbc_p = torch.zeros(n_obs, 7)

    out_mtvza = mtvza(obs_tb, model_profiles, surf_state, varbc_p)

    # 4. Оператор ИКФС-2
    obs_spec = torch.randn(n_obs, 180)
    t_prof = torch.randn(n_obs, 37, requires_grad=True)
    q_prof = torch.randn(n_obs, 37, requires_grad=True)
    t_skin = torch.randn(n_obs, 1, requires_grad=True)
    mu_sat = torch.ones(n_obs, 1)
    fcld = torch.zeros(n_obs)
    cth = torch.full((n_obs,), 1013.25)

    out_ikfs2 = ikfs2(obs_spec, t_prof, q_prof, t_skin, mu_sat, varbc_p, fcld, cth)

    # 5. AMV наблюдения
    amv_pred = torch.randn(4, 2, requires_grad=True)
    amv_obs = torch.randn(4, 2)
    amv_sigma = torch.full((4, 1), 2.0)
    amv_qi = torch.full((4, 1), 0.95)

    # 6. Робастный функционал потерь
    varbc_params = list(mtvza.varbc.parameters()) + list(ikfs2.varbc.parameters())
    varbc_priors = [torch.zeros_like(p) for p in varbc_params]

    loss_dict = loss_fn(out_mtvza, out_ikfs2, amv_pred, amv_obs, amv_sigma, amv_qi, varbc_params, varbc_priors)
    total_loss = loss_dict["loss_sat_total"] + h_modulated.sum() * 0.001
    total_loss.backward()

    # Проверка ненулевых градиентов
    assert raw_channels.grad is not None and torch.isfinite(raw_channels.grad).all()
    assert model_profiles.grad is not None and torch.isfinite(model_profiles.grad).all()
    assert t_prof.grad is not None and torch.isfinite(t_prof.grad).all()
    assert h_state.grad is not None and torch.isfinite(h_state.grad).all()

    return {
        "status": "assimilation_backward_verified",
        "loss_sat_total": float(loss_dict["loss_sat_total"].item()),
        "loss_mtvza": float(loss_dict["loss_mtvza"].item()),
        "loss_ikfs2": float(loss_dict["loss_ikfs2"].item()),
        "loss_amv": float(loss_dict["loss_amv"].item()),
        "gradient_norms": {
            "msugs_raw": float(torch.norm(raw_channels.grad).item()),
            "model_profiles": float(torch.norm(model_profiles.grad).item()),
            "ikfs2_t_prof": float(torch.norm(t_prof.grad).item()),
            "film_state": float(torch.norm(h_state.grad).item()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Подготовка контура ассимиляции спутников РФ")
    parser.add_argument("--output", type=Path, default=Path("outputs/satellite_assimilation_ready.json"))
    args = parser.parse_args()

    print("Инициализация спутникового контура ассимиляции...")
    pipeline = build_satellite_assimilation_pipeline()
    print("Верификация сквозного градиентного шага (Forward + Huber + VarBC + Backward)...")
    res = verify_assimilation_step(pipeline)
    print("Результаты верификации:", json.dumps(res, indent=2, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dict(res, timestamp=datetime.now(timezone.utc).isoformat()), f, indent=2, ensure_ascii=False)
    print(f"Готовность зафиксирована в {args.output}")


if __name__ == "__main__":
    main()
