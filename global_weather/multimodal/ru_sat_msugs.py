from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

R_EARTH_KM: float = 6371.0
H_ATM_KM: float = 8.4
R_GEO_KM: float = 42164.0
C1_PLANCK: float = 1.191042e-5  # мВт / (м^2 * ср * см^-4)
C2_PLANCK: float = 1.438775     # К * см


@dataclass(frozen=True)
class MSUGSChannelSpec:
    """Параметры 10 каналов МСУ-ГС / МСУ-ГС/М (Электро-Л и Арктика-М)."""
    vis_indices: tuple[int, ...] = (0, 1, 2)  # 0.5-0.65, 0.65-0.8, 0.8-0.9 мкм
    ir_indices: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9)  # 3.8, 6.4, 8.0, 8.7, 9.7, 10.7, 11.9 мкм
    ir_wavenumbers_cm1: tuple[float, ...] = (
        2631.58, 1562.50, 1250.00, 1149.43, 1030.93, 934.58, 840.34
    )
    band_a: tuple[float, ...] = (0.9985, 0.9991, 0.9993, 0.9994, 0.9992, 0.9996, 0.9997)
    band_b: tuple[float, ...] = (0.35, 0.18, 0.12, 0.10, 0.14, 0.08, 0.06)


class ArktikaElektroMSUGSAdapter(nn.Module):
    """
    Выполняет радиометрическую конверсию, расчет нестационарной геометрии ВЭО/ГСО,
    физических разностей каналов (BTD) и бесшовную сшивку на узлах икосаэдрической сетки mesh8.
    """

    def __init__(self, spec: MSUGSChannelSpec = MSUGSChannelSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.register_buffer("nu_c", torch.tensor(spec.ir_wavenumbers_cm1, dtype=torch.float32))
        self.register_buffer("band_a", torch.tensor(spec.band_a, dtype=torch.float32))
        self.register_buffer("band_b", torch.tensor(spec.band_b, dtype=torch.float32))

    @staticmethod
    def compute_viewing_geometry(
        node_xyz_km: torch.Tensor,      # [N_nodes, 3] в ECEF
        node_normal: torch.Tensor,      # [N_nodes, 3] единичная нормаль
        sat_xyz_km: torch.Tensor,       # [B, N_sats, 3] положение КА в момент t
        sun_dir_ecef: torch.Tensor,     # [B, 3] единичный вектор на Солнце
    ) -> dict[str, torch.Tensor]:
        """Расчет зенитных углов, оптической массы и расширения пикселя для ГСО и ВЭО."""
        # Вектор от узла сетки к КА: [B, N_sats, N_nodes, 3]
        d_vec = sat_xyz_km.unsqueeze(2) - node_xyz_km.unsqueeze(0).unsqueeze(1)
        r_sat = torch.linalg.norm(d_vec, dim=-1).clamp(min=100.0)  # [B, N_sats, N_nodes]

        # Косинус зенитного угла спутника mu_sat
        mu_sat = (d_vec * node_normal.unsqueeze(0).unsqueeze(1)).sum(dim=-1) / r_sat
        mu_sat_clamped = mu_sat.clamp(min=0.05, max=1.0)

        # Сферическая оптическая масса атмосферы
        eta = H_ATM_KM / R_EARTH_KM
        air_mass = (torch.sqrt(mu_sat_clamped**2 + 2.0 * eta + eta**2) - mu_sat_clamped) / eta

        # Коэффициент геометрического растяжения пятна относительно подспутниковой точки ГСО
        r_norm = r_sat / R_GEO_KM
        footprint_factor = (r_norm**2) / mu_sat_clamped

        # Косинус зенитного угла Солнца: [B, N_nodes]
        mu_sun = (node_normal.unsqueeze(0) * sun_dir_ecef.unsqueeze(1)).sum(dim=-1)

        return {
            "mu_sat": mu_sat,
            "air_mass": air_mass,
            "r_norm": r_norm,
            "footprint_factor": footprint_factor,
            "mu_sun": mu_sun,
        }

    def radiance_to_bt_and_brdf(
        self,
        raw_channels: torch.Tensor,     # [B, N_sats, N_nodes, 10]
        mu_sun: torch.Tensor,           # [B, N_nodes]
        gsics_slope: torch.Tensor,      # [N_sats, 7] кросс-калибровка по ИКФС-2
        gsics_offset: torch.Tensor,     # [N_sats, 7]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Перевод ВИД-каналов в BRDF и ИК-каналов в калиброванную яркостную температуру."""
        vis_raw = raw_channels[..., :3]
        ir_rad = raw_channels[..., 3:].clamp(min=1e-4)

        # Нормировка видимых каналов на косинус зенитного угла Солнца (маскирование ночи при mu_sun < 0.14)
        daylight_mask = (mu_sun > 0.14).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N_nodes, 1]
        mu_sun_safe = mu_sun.clamp(min=0.14).unsqueeze(1).unsqueeze(-1)
        vis_brdf = (vis_raw / mu_sun_safe) * daylight_mask

        # Обращение функции Планка для 7 ИК-каналов
        nu = self.nu_c.view(1, 1, 1, 7)
        t_star = (C2_PLANCK * nu) / torch.log1p((C1_PLANCK * (nu**3)) / ir_rad)
        tb_raw = (t_star - self.band_b.view(1, 1, 1, 7)) / self.band_a.view(1, 1, 1, 7)

        # Межспутниковая коррекция GSICS (приведение Электро-Л и Арктики-М к эталону ИКФС-2)
        tb_cal = tb_raw * gsics_slope.unsqueeze(0).unsqueeze(2) + gsics_offset.unsqueeze(0).unsqueeze(2)
        return torch.cat([vis_brdf, tb_cal], dim=-1), daylight_mask.squeeze(-1)

    def forward(
        self,
        raw_channels: torch.Tensor,     # [B, N_sats, N_nodes, 10]
        valid_mask: torch.Tensor,       # [B, N_sats, N_nodes] (включая фильтр рабочего участка ВЭО)
        node_xyz_km: torch.Tensor,      # [N_nodes, 3]
        node_normal: torch.Tensor,      # [N_nodes, 3]
        sat_xyz_km: torch.Tensor,       # [B, N_sats, 3]
        sun_dir_ecef: torch.Tensor,     # [B, 3]
        gsics_slope: torch.Tensor,      # [N_sats, 7]
        gsics_offset: torch.Tensor,     # [N_sats, 7]
    ) -> torch.Tensor:
        """
        Формирует 20-канальный композитный тензор признаков [B, N_nodes, 20] для узлов mesh8.
        """
        geom = self.compute_viewing_geometry(node_xyz_km, node_normal, sat_xyz_km, sun_dir_ecef)
        cal_ch, daylight = self.radiance_to_bt_and_brdf(
            raw_channels, geom["mu_sun"], gsics_slope, gsics_offset
        )

        # Расчет весов композитирования ГСО + ВЭО
        mu_sat = geom["mu_sat"]
        air_mass = geom["air_mass"]
        fp_factor = geom["footprint_factor"]

        limb_gate = torch.sigmoid(25.0 * (mu_sat - 0.3746))  # отсечка при theta_sat > 68 град
        raw_weights = (
            torch.exp(-0.35 * (air_mass - 1.0).pow(2) - 0.15 * (fp_factor - 1.0).pow(2))
            * limb_gate
            * valid_mask.float()
        )
        weight_sum = raw_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        norm_weights = raw_weights / weight_sum  # [B, N_sats, N_nodes]

        # Взвешенная свертка по всем КА (Электро-Л + Арктика-М)
        w_exp = norm_weights.unsqueeze(-1)
        fused_ch = (cal_ch * w_exp).sum(dim=1)  # [B, N_nodes, 10]

        # Информативные разности яркостных температур (BTD)
        # Индексы в fused_ch: 3->3.8мкм, 4->6.4мкм(WV), 8->10.7мкм, 9->11.9мкм
        btd_split = fused_ch[..., 8:9] - fused_ch[..., 9:10]     # 10.7 - 11.9 мкм (перистая облачность/влага)
        btd_wv = fused_ch[..., 4:5] - fused_ch[..., 8:9]         # 6.4 - 10.7 мкм (глубокая конвекция)
        btd_fog = fused_ch[..., 3:4] - fused_ch[..., 8:9]        # 3.8 - 10.7 мкм (туман / фаза облаков)

        # Эффективная геометрия сшитого наблюдения
        eff_mu_sat = (mu_sat * norm_weights).sum(dim=1).unsqueeze(-1)
        eff_air_mass = (air_mass * norm_weights).sum(dim=1).unsqueeze(-1)
        eff_r_norm = (geom["r_norm"] * norm_weights).sum(dim=1).unsqueeze(-1)
        mu_sun_feat = geom["mu_sun"].unsqueeze(-1)
        coverage_flag = (raw_weights.sum(dim=1) > 1e-3).float().unsqueeze(-1)
        multi_sat_overlap = ((raw_weights > 0.1).float().sum(dim=1) > 1.0).float().unsqueeze(-1)
        daylight_feat = daylight.squeeze(1).unsqueeze(-1) if daylight.dim() == 3 else daylight.unsqueeze(-1)

        return torch.cat(
            [
                fused_ch,           # 10 каналов
                btd_split,          # 1 канал
                btd_wv,             # 1 канал
                btd_fog,            # 1 канал
                eff_mu_sat,         # 1 канал
                eff_air_mass,       # 1 канал
                eff_r_norm,         # 1 канал
                mu_sun_feat,        # 1 канал
                coverage_flag,      # 1 канал
                multi_sat_overlap,  # 1 канал
                daylight_feat,      # 1 канал
            ],
            dim=-1,
        )


class ViewConditionedFiLMBlock(nn.Module):
    """
    Блок модуляции признаков (FiLM) для `global_weather/multimodal/networks.py`,
    компенсирующий остаточные угловые искажения на стыке ГСО и ВЭО.
    """

    def __init__(self, hidden_dim: int, geom_dim: int = 4) -> None:
        super().__init__()
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.feat_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, geom_features: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N_nodes, hidden_dim]
        geom_features: [B, N_nodes, 4] -> (eff_mu_sat, eff_air_mass, eff_r_norm, mu_sun)
        """
        gamma, beta = self.geom_mlp(geom_features).chunk(2, dim=-1)
        h_mod = self.norm(self.feat_proj(h)) * (1.0 + gamma) + beta
        return h + F.gelu(h_mod)
