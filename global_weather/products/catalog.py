"""Scientific identities of derived products, not interchangeable band names."""
from dataclasses import dataclass

SCHEMA = 'satellite-products-1'


@dataclass(frozen=True)
class ProductSpec:
    units: str
    vertical: str
    max_age_hours: int
    limits: tuple[float, float]
    methods: tuple[str, ...]
    title: str


# Maximum ages are explicit research admission policies, not validated skill horizons.
CATALOG = {
    'ndvi': ProductSpec('1', 'surface', 168, (-1., 1.), ('ndvi-surface-v1',), 'NDVI поверхности'),
    'ndvi_toa': ProductSpec('1', 'surface', 168, (-1., 1.), ('ndvi-toa-v1',), 'NDVI на верхней границе атмосферы'),
    'ndmi': ProductSpec('1', 'surface', 72, (-1., 1.), ('ndmi-surface-v1',), 'Водность растительности: индекс NDMI'),
    'ndsi': ProductSpec('1', 'surface', 24, (-1., 1.), ('ndsi-surface-v1',), 'Спектральный индекс снега NDSI'),
    'cloud_top_brightness_temperature': ProductSpec('K', 'column', 3, (100., 400.), ('cloud-ir-bt-v1',), 'Яркостная температура облачного пикселя'),
    'cloud_top_height': ProductSpec('m', 'column', 3, (-1000., 30000.), ('opaque-profile-height-v1',), 'Оценка высоты верхней границы облаков'),
    'cloud_liquid_water_path': ProductSpec('kg m-2', 'column', 3, (0., 20.), ('homogeneous-liquid-lwp-v1', 'reflectance-lut-liquid-lwp-v1'), 'Водозапас жидкой воды облака'),
    'soil_moisture_surface': ProductSpec('m3 m-3', 'surface', 24, (0., 1.), ('tau-omega-lut-v1',), 'Объёмная влажность поверхностного слоя почвы'),
    'land_surface_temperature': ProductSpec('K', 'surface', 6, (100., 400.), ('monochromatic-surface-temperature-v1',), 'Оценка температуры поверхности суши'),
}


def variable_spec(product, source, platform, *, history_hours=None, depth_bottom_m=None, method=None):
    spec = CATALOG[product]
    chosen = spec.methods[0] if method is None else method
    if chosen not in spec.methods: raise ValueError('Метод не соответствует продукту.')
    age = spec.max_age_hours if history_hours is None else history_hours
    return dict(units=spec.units, offset=0., scale=1., vertical=spec.vertical,
                source=source, platform=platform, channel_id=None, product=product,
                method=chosen, history_hours=age, product_depth_m=depth_bottom_m)
