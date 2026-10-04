"""Lazy per-example raster loading on top of the existing prepared-data contract."""
from dataclasses import dataclass
import numpy as np
import torch
from ..pipeline.dataset import PreparedDataset
from ..pipeline.io import artifact, MAX_JSON
from ..observations import PackedObservations, Variable, pack_observations
from ..vertical import PRESSURE_HPA
from .frames import sensor_registry, fingerprint, load_scene


@dataclass
class MultimodalObservations(PackedObservations):
    scenes: tuple = ()
    multimodal_fingerprint: str = ''

    def to(self, device):
        fields={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in vars(self).items()}
        fields['scenes']=tuple(scene.to(device) for scene in self.scenes)
        return MultimodalObservations(**fields)


class TrainingDataset(PreparedDataset):
    def __init__(self, path, *, require_raster_norm=True, **kwargs):
        super().__init__(path, **kwargs)
        raw=self.manifest.get('multimodal')
        self.multimodal = sensor_registry(raw) if raw is not None else None
        self._satellite = {s['id']:s.get('satellite',{}) for s in self.manifest['samples']}
        if self.multimodal is None:
            if any(self._satellite.values()):
                raise ValueError('Кадры объявлены без реестра приборов.')
            return
        raster_variables={ch['variable'] for s in self.multimodal['sensors'].values() for ch in s['channels']}
        if raster_variables & set(self.registry):
            raise ValueError('Один канал нельзя одновременно подать растром и тем же скалярным входом.')
        if self.norm is not None and require_raster_norm:
            for spec in self.multimodal['sensors'].values():
                for ch in spec['channels']:
                    self.norm.get(ch['variable'],ch['units']).at()
        for sample in self.samples:
            inputs=self._satellite[sample.id]
            if not isinstance(inputs,dict) or set(inputs)-set(self.multimodal['sensors']):
                raise ValueError('Кадр принадлежит неизвестному прибору.')
            for name,refs in inputs.items():
                if not isinstance(refs,list) or len(refs)>32:
                    raise ValueError('Допускается до 32 фрагментов на прибор и пример.')
                hashes=[]
                for ref in refs:
                    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
                        raise ValueError('У каждого кадра нужны путь и SHA256.')
                    hashes.append(ref['sha256'])
                if len(set(hashes))!=len(hashes):
                    raise ValueError('Один кадр повторён внутри примера.')

    def scenes(self, sample):
        if self.multimodal is None:
            return ()
        if self.norm is None:
            raise ValueError('До загрузки тензоров зафиксируйте нормы каналов.')
        output=[]; byte_count=0
        for name,refs in sorted(self._satellite[sample.id].items()):
            spec=self.multimodal['sensors'][name]
            for ref in refs:
                path=artifact(self.root,ref,limit=128*1024**2)
                scene=load_scene(path,spec,name,self.grid(),self.norm,sample.issue,self.kind,ref['sha256'])
                if scene is not None:
                    byte_count+=sum(t.numel()*t.element_size() for t in vars(scene).values() if isinstance(t,torch.Tensor))
                    if byte_count>128*1024**2:
                        raise ValueError('Кадры примера превышают 128 МиБ. Разбейте область на фрагменты.')
                    output.append(scene)
        return tuple(sorted(output,key=lambda s:(s.observed,s.sensor,s.identity)))

    def packed(self, sample):
        if self.multimodal is None:
            return super().packed(sample)
        variables={name:Variable(**spec) for name,spec in self.registry.items()}
        packed=pack_observations(self.eligible_records(sample),self.grid(),np.array(PRESSURE_HPA)*100,
                                 sample.issue,variables,normalization=self.norm)
        bad={k:v for k,v in packed.rejected.items() if k!='not_available_in_12h_window' and v}
        if bad:raise ValueError(f'Скалярные наблюдения не прошли допуск: {bad}')
        scenes=self.scenes(sample)
        if not packed.accepted_records and not scenes:
            raise ValueError('Нет пригодных наблюдений или спутниковых кадров.')
        values=vars(packed).copy()
        # This count now includes admitted radiometric scalars, not independent samples.
        values['accepted_records']+=sum(int(s.valid.sum()) for s in scenes)
        return MultimodalObservations(**values,scenes=scenes,
                                      multimodal_fingerprint=fingerprint(self.multimodal))

    def validate(self, **kwargs):
        report=super().validate(**kwargs)
        report['neural_input_path']='multimodal-v1' if self.multimodal else 'scalar-adaptive-v2'
        report['sensors']=list(self.multimodal['sensors']) if self.multimodal else []
        return report

    def input_activation_estimate(self):
        if self.multimodal is None:
            return 0
        maximum=0
        for sample in self.samples:
            if sample.split=='test':
                continue
            scenes=self.scenes(sample)
            amount=sum(scene.values.shape[1]*scene.values.shape[2]*(2*scene.values.shape[0]+8+128)*4*24
                       for scene in scenes)
            maximum=max(maximum,amount)
        return maximum
