"""Conditional VIS/SWIR cloud inversion with a sensor-specific radiative-transfer LUT.

Based on the two-channel principle of Nakajima & King, not on copying MODIS
coefficients. A fixed geometry/surface/atmosphere slice is required. Interpolation
is piecewise affine on triangles of the (tau, r_eff) grid. Multiple solutions,
extrapolation and poor sensitivity are rejected. No real LUT is bundled.
"""
from dataclasses import dataclass
import copy
import numpy as np
from .core import Field, QC, aligned, condition, result, canonical, require_hash, utc

CLOUD_CHECKS = frozenset({'liquid_phase', 'single_layer', 'fully_cloudy',
                          'plane_parallel_supported', 'surface_background_checked'})


@dataclass(frozen=True)
class CloudOpticsLUT:
    optical_depth: np.ndarray
    radius_m: np.ndarray
    reflectance: np.ndarray  # [tau, radius, 2]: VIS/NIR then absorbing SWIR
    metadata: dict
    source_sha256: str

    def __post_init__(self):
        t, r, f = (np.array(x, dtype=float, copy=True) for x in
                   (self.optical_depth, self.radius_m, self.reflectance))
        if (t.ndim != 1 or r.ndim != 1 or min(len(t), len(r)) < 3
                or len(t)*len(r) > 4096 or f.shape != (len(t), len(r), 2)):
            raise ValueError('Нужна таблица tau × re × два канала; не более 4096 узлов.')
        if (not all(np.isfinite(x).all() for x in (t, r, f))
                or not (np.diff(t) > 0).all() or not (np.diff(r) > 0).all()
                or t[0] <= 0 or t[-1] > 200 or r[0] <= 0 or r[-1] > 100e-6
                or (f < 0).any() or (f > 4).any()):
            raise ValueError('Неверные оси или отражение облачной таблицы.')
        m = copy.deepcopy(self.metadata)
        for key in ('source', 'platform', 'instrument', 'calibration_family',
                    'surface_condition_id', 'atmosphere_condition_id',
                    'radiative_transfer_model', 'reference', 'license'):
            if not isinstance(m.get(key), str) or not m[key].strip():
                raise ValueError(f'Не задано происхождение таблицы: {key}')
        if (m.get('data_kind') not in ('real', 'synthetic') or m.get('phase') != 'liquid'
                or m.get('vertical_model') != 'homogeneous' or m.get('quantity') != 'toa_reflectance'):
            raise ValueError('Нужна таблица отражения однородного жидкого облака.')
        ids = m.get('channel_ids')
        if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or not all(isinstance(x,str) and x for x in ids):
            raise ValueError('Требуются два разных физических канала.')
        bands = m.get('wavelength_um')
        if (not isinstance(bands, list) or len(bands) != 2
                or not all(type(v) in (int,float) and np.isfinite(v) for v in bands)
                or not 0.4 <= bands[0] <= 1.3 or not 1.5 <= bands[1] <= 2.5):
            raise ValueError('Нужны VIS/NIR и поглощающая полоса 1,5–2,5 мкм. Канал 3,7 мкм не поддержан.')
        responses = m.get('spectral_response_sha256')
        if not isinstance(responses, list) or len(responses) != 2:
            raise ValueError('Нужны хэши спектральных характеристик обоих каналов.')
        for value in responses: require_hash(value)
        for key, upper in (('solar_zenith_deg',80),('view_zenith_deg',80),('relative_azimuth_deg',180)):
            if type(m.get(key)) not in (int,float) or not 0 <= m[key] <= upper:
                raise ValueError('Неверная геометрия таблицы.')
        if type(m.get('geometry_tolerance_deg')) not in (int,float) or not 0 < m['geometry_tolerance_deg'] <= .5:
            raise ValueError('Явный допуск геометрии должен быть в (0; 0,5] градуса.')
        require_hash(self.source_sha256); canonical(m)
        for x in (t,r,f): x.setflags(write=False)
        object.__setattr__(self,'optical_depth',t)
        object.__setattr__(self,'radius_m',r)
        object.__setattr__(self,'reflectance',f)
        object.__setattr__(self,'metadata',m)


@dataclass(frozen=True)
class OpticalSolution:
    optical_depth: np.ndarray
    radius_m: np.ndarray
    covariance: np.ndarray  # [...,2,2], parameters are tau [1], re [m]
    valid: np.ndarray
    qc: np.ndarray


def _triangles(lut):
    coordinates = np.stack(np.meshgrid(lut.optical_depth,lut.radius_m,indexing='ij'),-1)
    rows, cols = coordinates.shape[:2]
    ids=[]
    for i in range(rows-1):
        for j in range(cols-1):
            a,b,c,d = i*cols+j,(i+1)*cols+j,(i+1)*cols+j+1,i*cols+j+1
            ids.extend(((a,b,c),(a,c,d)))
    ids = np.asarray(ids)
    parameters=coordinates.reshape(-1,2)[ids]
    reflectance=lut.reflectance.reshape(-1,2)[ids]
    # Columns are differences to the first triangle vertex.
    b = np.swapaxes(reflectance[:,1:]-reflectance[:,:1],1,2)
    determinant = np.linalg.det(b)
    good = np.abs(determinant) > 1e-14
    if not good.any(): return None
    p=parameters[good]; f=reflectance[good]
    inv=np.linalg.inv(b[good])
    a=np.swapaxes(p[:,1:]-p[:,:1],1,2)
    derivative=a@inv  # dx/dR, includes correlated parameter errors
    return p,f,inv,derivative


def retrieve_cloud_optics(nonabsorbing: Field, absorbing: Field, solar_zenith: Field,
                          view_zenith: Field, relative_azimuth: Field, eligible: Field,
                          lut: CloudOpticsLUT, *, available_at, data_kind,
                          max_condition=1e4, max_relative_sigma=1.):
    """Invert calibrated paired reflectances, retaining full conditional covariance.

    Two channels retrieve two parameters: a zero residual is NOT independent
    goodness-of-fit evidence. Noise correlations, atmospheric/3D/cloud-phase and
    LUT interpolation errors are not included in the reported covariance.
    """
    if (data_kind not in ('real','synthetic') or lut.metadata['data_kind'] != data_kind
            or not np.isfinite([max_condition,max_relative_sigma]).all()
            or max_condition <= 1 or max_relative_sigma <= 0):
        raise ValueError('Неверное происхождение или пороги облачной инверсии.')
    inputs=[nonabsorbing,absorbing,solar_zenith,view_zenith,relative_azimuth,eligible]
    m=lut.metadata
    for k,f in enumerate((nonabsorbing,absorbing)):
        f.require('toa_reflectance','1')
        if f.uncertainty is None: raise ValueError('Нужна погрешность обоих отражательных каналов.')
        for key in ('source','platform','instrument','calibration_family',
                    'surface_condition_id','atmosphere_condition_id'):
            if f.metadata.get(key) != m[key]: raise ValueError('Таблица не соответствует прибору, калибровке или условиям.')
        if (f.metadata.get('channel_id') != m['channel_ids'][k]
                or f.metadata.get('wavelength_um') != m['wavelength_um'][k]
                or f.metadata.get('spectral_response_sha256') != m['spectral_response_sha256'][k]):
            raise ValueError('Не совпадают физический канал или спектральная характеристика.')
    solar_zenith.require('solar_zenith_angle','degree')
    view_zenith.require('view_zenith_angle','degree')
    relative_azimuth.require('relative_azimuth_angle','degree')
    if not CLOUD_CHECKS.issubset(set(eligible.metadata.get('checks',[]))):
        raise ValueError('Нужны проверки фазы, слоя, заполнения пикселя и фона.')
    if data_kind=='real' and any(f.metadata.get('data_kind')!='real' for f in inputs):
        raise ValueError('Синтетический вход не допускается как реальный.')
    if any(utc(f.available_at)>utc(available_at) for f in inputs):
        raise ValueError('Продукт не может быть готов раньше входов.')
    valid=np.array(aligned(*inputs,max_skew_seconds=60.),dtype=bool,copy=True)
    flags=np.where(valid,0,int(QC.MISSING)).astype(np.uint16)
    ok=condition(eligible,nonabsorbing)
    for key,f in (('solar_zenith_deg',solar_zenith),('view_zenith_deg',view_zenith),('relative_azimuth_deg',relative_azimuth)):
        upper = 180. if key == 'relative_azimuth_deg' else 80.
        ok &= (f.values >= 0) & (f.values <= upper) & (np.abs(f.values-m[key])<=m['geometry_tolerance_deg'])
    flags[~ok]|=int(QC.CONDITIONS)
    noise=np.stack([nonabsorbing.uncertainty,absorbing.uncertainty],-1)
    measured=np.stack([nonabsorbing.values,absorbing.values],-1)
    domain=(np.isfinite(noise)&(noise>0)).all(-1) & ((measured>=0)&(measured<=4)).all(-1)
    flags[~domain]|=int(QC.DOMAIN)
    valid &= ok & domain
    out=np.full((*valid.shape,2),np.nan)
    cov=np.full((*valid.shape,2,2),np.nan)
    triangles=_triangles(lut)
    if triangles is None:
        flags[valid]|=int(QC.LOW_SENSITIVITY); valid[...]=False
        return OpticalSolution(out[...,0],out[...,1],cov,valid,flags)
    p,f,inv,dxdr=triangles
    ranges=np.array([np.ptp(lut.optical_depth),np.ptp(lut.radius_m)])
    lower=np.array([lut.optical_depth[0],lut.radius_m[0]])
    upper=lower+ranges
    for index in np.ndindex(valid.shape):
        if not valid[index]: continue
        bary=np.einsum('tij,tj->ti',inv,measured[index]-f[:,0])
        inside=(bary>=-1e-10).all(1)&(bary.sum(1)<=1+1e-10)
        candidates=np.flatnonzero(inside)
        if not len(candidates):
            valid[index]=False; flags[index]|=int(QC.NO_SOLUTION); continue
        x=p[candidates,0]+np.einsum('tij,ti->tj',p[candidates,1:]-p[candidates,:1],bary[candidates])
        if np.any(np.linalg.norm((x-x[0])/ranges,axis=1)>1e-6):
            valid[index]=False; flags[index]|=int(QC.AMBIGUOUS); continue
        chosen=candidates[0]; estimate=x[0]
        scaled=dxdr[chosen]*noise[index][None,:]/ranges[:,None]
        c=(dxdr[chosen]*noise[index][None,:])@(dxdr[chosen]*noise[index][None,:]).T
        sigma=np.sqrt(np.maximum(np.diag(c),0.))
        if (not np.isfinite(c).all() or np.linalg.cond(scaled)>max_condition
                or (sigma/estimate>max_relative_sigma).any()):
            valid[index]=False; flags[index]|=int(QC.LOW_SENSITIVITY); continue
        if ((estimate-lower)/ranges<=1e-8).any() or ((upper-estimate)/ranges<=1e-8).any():
            valid[index]=False; flags[index]|=int(QC.BOUNDARY); continue
        out[index],cov[index]=estimate,c
    return OpticalSolution(out[...,0],out[...,1],cov,valid,flags)


def cloud_lwp_from_reflectances(nonabsorbing, absorbing, solar_zenith, view_zenith,
                               relative_azimuth, eligible, lut, *, available_at,
                               data_kind, max_condition=1e4, max_relative_sigma=1.):
    """Paired reflectances -> tau/re -> LWP, retaining conditional covariance."""
    solution=retrieve_cloud_optics(nonabsorbing,absorbing,solar_zenith,view_zenith,
        relative_azimuth,eligible,lut,available_at=available_at,data_kind=data_kind,
        max_condition=max_condition,max_relative_sigma=max_relative_sigma)
    t,r,c=solution.optical_depth,solution.radius_m,solution.covariance
    factor=2*1000./3
    value=factor*t*r
    variance=factor**2*(r*r*c[...,0,0]+t*t*c[...,1,1]+2*t*r*c[...,0,1])
    uncertainty=np.sqrt(np.maximum(variance,0))
    return result('cloud_liquid_water_path','reflectance-lut-liquid-lwp-v1',
        value,solution.valid,solution.qc,primary=nonabsorbing,
        dependencies=[nonabsorbing,absorbing,solar_zenith,view_zenith,relative_azimuth,eligible],
        available_at=available_at,source=nonabsorbing.metadata['source'],
        platform=nonabsorbing.metadata['platform'],data_kind=data_kind,uncertainty=uncertainty,
        assumptions=['plane_parallel_homogeneous_liquid_layer','sensor_response_specific_lut',
            'piecewise_affine_parameter_grid','fixed_surface_atmosphere_and_angles',
            'independent_reflectance_noise_correlated_retrieved_parameters',
            'conditional_error_not_total_error','zero_residual_not_validation',
            'no_ice_water_or_vapour_retrieval'],
        attributes={'lut_sha256':lut.source_sha256,'lut_provenance':lut.metadata,
                    'parameter_covariance_used':True,'max_condition':max_condition,
                    'max_relative_sigma':max_relative_sigma})
