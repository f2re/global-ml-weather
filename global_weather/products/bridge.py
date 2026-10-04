"""Read existing Arctic/Electro numerical capsules without reapplying calibration."""
from pathlib import Path
import numpy as np
from .core import Field, digest
from .io import save_field, exclusive_bytes, sha256


def from_capsule(directory, output, *, data_kind, geometry_output=None):
    from ..connectors.raster_bridge import read_capsule
    if data_kind not in ('real','synthetic'): raise ValueError('Укажите происхождение данных.')
    meta,a = read_capsule(directory)
    if meta.get('data_kind') is not None and meta['data_kind'] != data_kind:
        raise ValueError('Запрещено изменять происхождение исходной капсулы.')
    if meta.get('quantity')!='brightness_temperature' or meta.get('units')!='K' or meta.get('source') not in ('arktika_m','electro_l'):
        raise ValueError('Мост поддерживает численные ИК-каналы, не RGB или отражение.')
    calibration = meta.get('calibration_reference')
    if not isinstance(calibration,str) or not calibration.strip(): raise ValueError('Нет происхождения калибровки.')
    grid_id=digest({'crs':meta['crs'],'transform':meta['transform'],'shape':meta['shape']})
    original=Path(directory)/'pixels.npz'; initial_hash=sha256(original)
    if initial_hash != meta['arrays_sha256']: raise ValueError('Капсула изменилась после чтения.')
    info=dict(source=meta['source'],platform=meta['platform'],instrument=meta['instrument'],
              channel_id=meta['channel_id'],calibration_reference=calibration,
              data_kind=data_kind,spectral_role='thermal_window' if str(meta['channel_id']) in ('9','10') else 'other_infrared',
              upstream_capsule_sha256=sha256(Path(directory)/'manifest.json'),
              time_support=meta.get('time_support'))
    field=Field(a['values'],a['valid'],'brightness_temperature','K',grid_id,meta['observed_at'],
                meta['available_at'],initial_hash,info)
    if geometry_output is not None:
        needed={'latitude','longitude','view_zenith_deg','footprint_km'}
        if not needed.issubset(a) or not meta.get('geometry_reference'):
            raise ValueError('Нет проверенной геометрии; нулевой угол не подставляется.')
        if Path(geometry_output).exists(): raise FileExistsError('Геометрия не перезаписывается.')
    save_field(output,field)
    if geometry_output is not None:
        g={k:a[k] for k in needed};g['grid_id']=np.array(grid_id)
        exclusive_bytes(geometry_output,lambda f:np.savez_compressed(f,**g))
    if sha256(original)!=initial_hash: raise ValueError('Капсула изменилась.')
    return dict(status='physical_field_exported',quantity=field.quantity,sha256=sha256(output),
                grid_id=grid_id,data_kind=data_kind,
                note='Шкала повторно не применяется. Маска облаков и остальные входы метода нужны отдельно.')
