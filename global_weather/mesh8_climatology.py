"""Compact seasonal context adapter for the mesh-8 qualification experiment.

NCEP1 monthly means are much coarser than the 655362-cell model mesh.  Storing
12 x 655362 x 37 x 5 pre-interpolated values would create a multi-gigabyte
artifact without adding source information.  This adapter therefore keeps the
immutable seasonal artifact on reviewed mesh level 4 and maps each fine model
cell to its nearest climate-context cell at sampling time.

The mapping is a network context operator, not an observation operator and not
a claim that NCEP1 has mesh-8 spatial resolution.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .grid import build_grid
from .seasonal_climatology import SeasonalClimatology

SOURCE_CONTEXT_LEVEL = 4
TARGET_MODEL_LEVEL = 8
ALGORITHM = 'mesh8-seasonal-nearest-context-v1'


class Mesh8SeasonalClimatology:
    def __init__(self, root, grid):
        if grid.level != TARGET_MODEL_LEVEL:
            raise ValueError('Mesh8SeasonalClimatology requires model mesh level 8.')
        source = SeasonalClimatology(root)
        if source.payload['mesh_level'] != SOURCE_CONTEXT_LEVEL:
            raise ValueError('Mesh-8 qualification requires a seasonal context artifact prepared on mesh level 4.')
        source_grid = build_grid(SOURCE_CONTEXT_LEVEL)
        mapping = np.asarray(source_grid.tree.query(grid.xyz)[1], dtype=np.int32)
        if mapping.shape != (grid.n_cells,) or np.any((mapping < 0) | (mapping >= source_grid.n_cells)):
            raise ValueError('Invalid coarse-to-mesh8 seasonal mapping.')
        self.source = source
        self.target_grid_fingerprint = grid.fingerprint
        self.mapping = mapping
        h = hashlib.sha256()
        h.update(ALGORITHM.encode())
        h.update(source.fingerprint.encode())
        h.update(grid.fingerprint.encode())
        h.update(mapping.astype('<i4', copy=False).tobytes())
        self.fingerprint = h.hexdigest()

    @property
    def payload(self):
        return {
            'schema': ALGORITHM,
            'source_climatology': self.source.payload,
            'source_mesh_level': SOURCE_CONTEXT_LEVEL,
            'target_mesh_level': TARGET_MODEL_LEVEL,
            'target_grid_fingerprint': self.target_grid_fingerprint,
            'mapping': 'nearest climate-context cell centre on the sphere',
            'scientific_acceptance': False,
        }

    def verify_sources(self):
        self.source.verify_sources()

    def sample(self, valid_time):
        mean, support = self.source.sample(valid_time)
        sampled_mean = np.asarray(mean[self.mapping], dtype=np.float32)
        sampled_support = np.asarray(support[self.mapping], dtype=bool)
        if (sampled_mean.shape != (len(self.mapping), 37, 5)
                or sampled_support.shape != sampled_mean.shape
                or not np.isfinite(sampled_mean[sampled_support]).all()):
            raise ValueError('Remapped mesh-8 seasonal context differs.')
        return sampled_mean, sampled_support
