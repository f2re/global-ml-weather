"""One connected observation-to-72h network, not an ensemble of disconnected demos."""
from datetime import timedelta
import torch
from torch import nn
from ..adaptive import AdaptiveWeatherModel, AnalysisState
from ..products.ingest import evidence_history
from .frames import sensor_registry, fingerprint
from .networks import SparseObservationEncoder, SatelliteEncoder, SourceFusion


class MultimodalWeatherModel(AdaptiveWeatherModel):
    def __init__(self, grids, vocabulary, *, multimodal, **kwargs):
        super().__init__(grids,vocabulary,**kwargs)
        self.multimodal=sensor_registry(multimodal)
        self.multimodal_fingerprint=fingerprint(multimodal)
        self.encoder=SparseObservationEncoder(len(vocabulary),self.hidden,grids[0])
        self.satellites=nn.ModuleDict({name:SatelliteEncoder(spec,self.hidden)
                                      for name,spec in sorted(multimodal['sensors'].items())})
        self.fusion=SourceFusion(self.hidden,len(self.satellites))

    def get_extra_state(self):
        return {**super().get_extra_state(),'architecture':'multimodal-v1',
                'multimodal':self.multimodal_fingerprint}

    def _validate_inputs(self, obs, elevation, land):
        super()._validate_inputs(obs,elevation,land)
        if getattr(obs,'multimodal_fingerprint',None)!=self.multimodal_fingerprint:
            raise ValueError('Схема многоканальных наблюдений отличается от весов.')

    def analysis_state(self, obs, elevation, land, background=None):
        initial=super().analysis_state(obs,elevation,land,background)
        previous=dict(background.evidence) if background is not None else {}
        fresh=[s for s in obs.scenes if s.identity not in previous]
        if not fresh:return initial
        features=[]; masks=[]
        for name,encoder in self.satellites.items():
            value,present=encoder([s for s in fresh if s.sensor==name],len(self.xyz))
            features.append(value);masks.append(present)
        masks=torch.stack(masks,1)
        state=self.fusion(initial.latent,torch.stack(features,1),masks)
        if bool(masks.any()):
            state=self._process(state,elevation,land,obs.issue_time)
        evidence=dict(initial.evidence)
        for scene in fresh:evidence[scene.identity]=scene.observed
        retained=tuple(sorted((key,time) for key,time in evidence.items()
                        if obs.issue_time-timedelta(hours=evidence_history(key))<time<=obs.issue_time))
        return AnalysisState(state,obs.issue_time,self.get_extra_state(),retained)


def make_model(ds,cfg):
    from ..grid import build_pyramid
    observations=ds.packed(ds.samples[0])
    options=dict(observation_schema=observations.schema_fingerprint,hidden=cfg.hidden,
                 latent_slots=cfg.latent_slots,step_hours=ds.step,normalization=ds.norm)
    grids=build_pyramid(ds.level)
    if getattr(ds,'multimodal',None) is None:
        if ds.manifest.get('multimodal'):
            raise ValueError('Используйте TrainingDataset для многоканальных входов.')
        return AdaptiveWeatherModel(grids,observations.vocabulary,**options)
    return MultimodalWeatherModel(grids,observations.vocabulary,multimodal=ds.multimodal,**options)


def describe(model):
    return {'architecture':model.get_extra_state()['architecture'],
            'parameters':sum(p.numel() for p in model.parameters()),
            'blocks':{name:sum(p.numel() for p in module.parameters())
                       for name,module in model.named_children()},
            'output_levels':len(model.pressure_pa),'forecast_step_hours':model.step_hours,
            'input_networks_connected':True,'meteorological_skill_verified':False}
