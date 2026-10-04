"""Связь новых кодировщиков с существующим физически адаптивным ядром и выходами."""
from __future__ import annotations
from dataclasses import replace
from datetime import timedelta
import json
import torch
from ..adaptive import AdaptiveWeatherModel, AnalysisState
from .contracts import utc
from .integration import MultimodalObservations
from .fusion import SparseObservationEncoder, SatelliteBank, SourceFusion


def frame_identity(sensor,key):
    return json.dumps(['multimodal-frame',sensor,key,'v1'],separators=(',',':'))


class MultimodalWeatherModel(AdaptiveWeatherModel):
    def __init__(self,*args,sensors,sensor_signature,radius_km=500.,neighbors=16,base_channels=16,**kwargs):
        for sensor in sensors: sensor.require_normalization()
        super().__init__(*args,**kwargs)
        self.sensors=tuple(sensors);self.sensor_signature=sensor_signature
        self.multimodal_settings={'radius_km':radius_km,'neighbors':neighbors,'base_channels':base_channels}
        self.encoder=SparseObservationEncoder(len(self.vocabulary),self.hidden,self.xyz,
                                              radius_km=radius_km,neighbors=neighbors)
        self.satellite_bank=SatelliteBank(self.sensors,self.hidden,base_channels)
        self.source_fusion=SourceFusion(self.hidden)
        self.output_grid=args[0][0]

    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(architecture='multimodal-adaptive-v1',sensor_signature=self.sensor_signature,
                     sensor_contracts=[s.signature for s in self.sensors],**self.multimodal_settings)
        return state

    def analysis_state(self,obs,elevation,land,background=None):
        if not isinstance(obs,MultimodalObservations) or obs.sensor_signature!=self.sensor_signature:
            raise ValueError('Многомодальный вход и схема модели не совпадают.')
        available=[];old=dict(background.evidence) if background else {}
        by_id={s.id:s for s in self.sensors}
        for seq in obs.sequences:
            if seq.sensor_id not in by_id:raise ValueError('Неизвестный прибор.')
            seq=seq.causal(obs.issue_time,by_id[seq.sensor_id])
            if seq is None:continue
            keep=torch.tensor([frame_identity(seq.sensor_id,key) not in old for key in seq.frame_ids],
                              device=seq.values.device,dtype=torch.bool)
            if not keep.any():continue
            changed={k:getattr(seq,k)[keep] for k in ('values','valid','observed_unix','available_unix',
                        'view_zenith_deg','solar_zenith_deg','footprint_km')}
            changed['frame_ids']=tuple(k for k,yes in zip(seq.frame_ids,keep.tolist()) if yes)
            available.append(replace(seq,**changed))
        analysed=super().analysis_state(obs,elevation,land,background)
        fields,masks=self.satellite_bank(available,obs.issue_time,self.output_grid)
        state=self.source_fusion(analysed.latent,fields,masks)
        if masks and any(m.any() for m in masks):
            state=self._process(state,elevation,land,obs.issue_time)
        evidence=dict(analysed.evidence)
        for seq in available:
            sensor=by_id[seq.sensor_id]
            _,valid=seq.normalized(sensor)
            for key,time,use in zip(seq.frame_ids,seq.observed_unix.tolist(),valid.flatten(1).any(1).tolist()):
                if use:
                    from datetime import datetime,timezone
                    evidence[frame_identity(seq.sensor_id,key)]=datetime.fromtimestamp(time,tz=timezone.utc)
        return AnalysisState(state,obs.issue_time,self.get_extra_state(),tuple(sorted(evidence.items())))
