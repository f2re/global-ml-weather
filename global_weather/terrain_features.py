"""Физические признаки и операторы глобального рельефа (DEM).

Обеспечивает точную выборку высоты из проверенного GMT Earth Relief 30s NetCDF,
вычисление топографического градиента (уклона) по сферическому графу,
расчёт барометрического гидростатического фактора и гидростатическую привязку
приземного давления к давлению на уровне моря.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn

from .grid import SphereGrid, latlon
from .terrain import dem_samples, local_file, verify_dem

# Физические постоянные стандартной атмосферы
G0 = 9.80665          # Ускорение свободного падения, м/с²
RD = 287.05           # Удельная газовая постоянная сухого воздуха, Дж/(кг·К)
T0 = 288.15           # Стандартная приземная температура, К
GAMMA = 0.0065        # Стандартный вертикальный градиент температуры, К/м
H_SCALE = 5000.0      # Инженерный масштаб нормировки высоты рельефа, м


def compute_terrain_slope(grid: SphereGrid, elevation_m: np.ndarray) -> np.ndarray:
    """Вычисляет средний локальный уклон рельефа |grad z| по смежным ячейкам сферического графа."""
    n = grid.n_cells
    src, dst = grid.edges[0], grid.edges[1]
    diffs = np.abs(elevation_m[dst] - elevation_m[src])

    # Расстояние по хорде сферы (радиус Земли ~ 6371 км)
    chord_dist = np.linalg.norm(grid.xyz[dst] - grid.xyz[src], axis=-1) * 6371000.0
    chord_dist = np.maximum(chord_dist, 1000.0) # минимум 1 км
    edge_slopes = diffs / chord_dist

    # Усреднение уклонов по рёбрам, входящим в каждую ячейку
    slope = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    np.add.at(slope, dst, edge_slopes)
    np.add.at(counts, dst, 1.0)
    return slope / np.maximum(counts, 1.0)


def barometric_pressure_ratio(elevation_m: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Вычисляет барометрическое отношение p_sfc / MSLP по гипсометрической формуле стандартной атмосферы."""
    exponent = G0 / (RD * GAMMA)
    if isinstance(elevation_m, torch.Tensor):
        base = torch.clamp(1.0 - (GAMMA * elevation_m) / T0, min=0.1)
        return torch.pow(base, exponent)
    base = np.clip(1.0 - (GAMMA * elevation_m) / T0, 0.1, None)
    return np.power(base, exponent)


def extract_grid_terrain(
    dem_path: str | Path,
    manifest_path: str | Path,
    grid: SphereGrid,
    reference_static: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Извлекает физические поля рельефа (высота в м, доля суши, уклон) для сетки модели."""
    dem_p = local_file(dem_path)
    manifest = json.loads(local_file(manifest_path).read_text())
    ll = latlon(grid.xyz)

    # Блочная выборка высоты из NetCDF без загрузки 896 МБ в память
    samples = dem_samples(dem_p, manifest, ll[:, 0], ll[:, 1])

    # Доля суши
    if reference_static is not None and Path(reference_static).exists():
        ref_data = np.load(reference_static)
        if 'land_fraction' in ref_data and len(ref_data['land_fraction']) == grid.n_cells:
            land_fraction = ref_data['land_fraction'].astype(np.float32)
        else:
            land_fraction = (samples > 0.0).astype(np.float32)
    else:
        # Для суши без отдельного статического файла: высоты выше 0 над уровнем геоида
        land_fraction = (samples > 0.0).astype(np.float32)

    slope = compute_terrain_slope(grid, samples)
    return samples.astype(np.float32), land_fraction, slope.astype(np.float32)


class OrographicCoupling(nn.Module):
    """Связывает приземное гидростатическое давление с давлением на уровне моря и высотой рельефа."""

    def __init__(self, hidden: int):
        super().__init__()
        self.barometric_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2), # Поправки к p_sfc и MSLP
        )

    def forward(
        self,
        predicted_mslp: Tensor,
        elevation_m: Tensor,
        temperature_2m: Tensor,
    ) -> Tensor:
        """Вычисляет физически обоснованное приземное гидростатическое давление."""
        # Барометрический фактор
        ratio = barometric_pressure_ratio(elevation_m)
        p_hydrostatic = predicted_mslp * ratio
        return p_hydrostatic
