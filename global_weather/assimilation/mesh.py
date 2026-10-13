"""Latent icosahedral mesh, multimesh graph and spherical interpolation operators.

The processor graph keeps the directed edges of EVERY refinement level 0..L
on the finest node set (GraphCast multimesh). Level-0 edges are ~7000 km
long, so one message-passing layer already connects distant regions; the
domain of dependence of one 3-hour step is no longer limited to one ring of
neighbours. Coarse vertices are the first 10*4**l+2 vertices of the finer
mesh by construction of the refinement, which is verified explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..grid import EARTH_RADIUS_M, _directed_edges, _topology, build_grid, grid_cell_count, unit_xyz
from .geometry import dual_geometry


def local_frames(xyz: np.ndarray) -> np.ndarray:
    """Return [N,3,3] rows (east, north, up) of the local tangent frame."""
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), -1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), -1)
    return np.stack((east, north, xyz), 1)


def multimesh_edges(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed (sender, receiver) union of refinement levels 0..level and per-edge level."""
    finest, _ = _topology(level)
    senders, receivers, levels = [], [], []
    for coarse in range(level + 1):
        xyz, faces = _topology(coarse)
        n = grid_cell_count(coarse)
        if xyz.shape != (n, 3) or not np.array_equal(xyz, finest[:n]):
            raise RuntimeError('Coarse vertices are not a prefix of the finest mesh.')
        edges = _directed_edges(faces)
        senders.append(edges[0]); receivers.append(edges[1]); levels.append(np.full(edges.shape[1], coarse))
    pairs = np.stack((np.concatenate(senders), np.concatenate(receivers)), 1)
    edge_level = np.concatenate(levels)
    unique, index = np.unique(pairs, axis=0, return_index=True)
    return unique.T.astype(np.int64), edge_level[index].astype(np.int64)


def edge_features(xyz: np.ndarray, senders: np.ndarray, receivers: np.ndarray, scale: float) -> np.ndarray:
    """GraphCast-style features: chord length and sender offset in the receiver frame, / scale."""
    frames = local_frames(xyz)
    offset = xyz[senders] - xyz[receivers]
    local = np.einsum('eij,ej->ei', frames[receivers], offset)
    length = np.linalg.norm(offset, axis=1, keepdims=True)
    return np.concatenate((length, local), 1) / scale


@dataclass
class Interpolation:
    """Spherical-triangle barycentric weights of query points on mesh nodes."""
    nodes: np.ndarray    # [P,3] int64
    weights: np.ndarray  # [P,3] float64, nonnegative, sum to one


class LatentMesh:
    """Geometry shared by the encoder, processor, decoder and diagnostics."""

    def __init__(self, level: int, *, multimesh: bool = True):
        if type(level) is not int or not 1 <= level <= 7:
            raise ValueError('Latent mesh level must be an integer within 1..7.')
        self.level = level
        self.grid = build_grid(level)
        self.xyz = self.grid.xyz.astype(np.float64)
        if multimesh:
            edges, edge_level = multimesh_edges(level)
        else:
            edges = self.grid.edges.astype(np.int64)
            edge_level = np.full(edges.shape[1], level)
        self.senders, self.receivers = edges
        self.edge_level = edge_level
        chords = np.linalg.norm(self.xyz[self.senders] - self.xyz[self.receivers], axis=1)
        self.edge_scale = float(chords.max())
        self.edge_features = edge_features(self.xyz, self.senders, self.receivers, self.edge_scale)
        fine = self.grid.edges
        fine_chord = np.linalg.norm(self.xyz[fine[0]] - self.xyz[fine[1]], axis=1)
        self.node_spacing_m = float(EARTH_RADIUS_M * np.mean(2 * np.arcsin(fine_chord / 2)))
        self.max_edge_m = float(EARTH_RADIUS_M * 2 * np.arcsin(min(self.edge_scale / 2, 1.0)))
        self.frames = local_frames(self.xyz)
        self.dual = dual_geometry(self.grid)
        self.areas_m2 = self.dual['areas_m2']
        self.tree = cKDTree(self.xyz)
        faces = np.asarray(self.grid.faces, dtype=np.int64)
        centroids = self.xyz[faces].mean(1)
        self.faces = faces
        self.face_tree = cKDTree(centroids / np.linalg.norm(centroids, axis=1, keepdims=True))

    @property
    def n_nodes(self) -> int:
        return len(self.xyz)

    def reach_km(self, hops: int) -> float:
        """Great-circle distance reachable from node 0 in `hops` processor layers."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        n = self.n_nodes
        adjacency = csr_matrix((np.ones(len(self.senders)), (self.senders, self.receivers)), shape=(n, n))
        distance = shortest_path(adjacency, unweighted=True, indices=0, directed=True)
        reached = distance <= hops
        angle = np.arccos(np.clip(self.xyz @ self.xyz[0], -1, 1))
        return float(EARTH_RADIUS_M * angle[reached].max() / 1000)

    def interpolation(self, latitude, longitude, candidates: int = 8) -> Interpolation:
        """Barycentric weights from the spherical triangle containing each point.

        Weights are the normalised triple-product ratios det(p,b,c)/det(a,b,c),
        i.e. barycentric coordinates of the gnomonic projection onto the face.
        """
        points = unit_xyz(np.asarray(latitude, float), np.asarray(longitude, float)).reshape(-1, 3)
        k = min(candidates, len(self.faces))
        _, face_index = self.face_tree.query(points, k=k)
        face_index = np.atleast_2d(face_index)
        nodes = np.zeros((len(points), 3), dtype=np.int64)
        weights = np.full((len(points), 3), np.nan)
        for column in range(k):
            pending = np.isnan(weights[:, 0])
            if not pending.any():
                break
            face = self.faces[face_index[pending, column]]
            a, b, c = (self.xyz[face[:, i]] for i in range(3))
            p = points[pending]
            whole = np.einsum('ij,ij->i', a, np.cross(b, c))
            raw = np.stack((np.einsum('ij,ij->i', p, np.cross(b, c)),
                            np.einsum('ij,ij->i', a, np.cross(p, c)),
                            np.einsum('ij,ij->i', a, np.cross(b, p))), 1) / whole[:, None]
            inside = (raw >= -1e-10).all(1)
            rows = np.flatnonzero(pending)[inside]
            nodes[rows] = face[inside]
            clipped = np.clip(raw[inside], 0, None)
            weights[rows] = clipped / clipped.sum(1, keepdims=True)
        missing = np.isnan(weights[:, 0])
        if missing.any():
            raise RuntimeError('Point outside every candidate spherical triangle.')
        return Interpolation(nodes, weights)

    def neighbours(self, latitude, longitude, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """k nearest nodes, great-circle distance [m] and offset in each node's frame / spacing."""
        points = unit_xyz(np.asarray(latitude, float), np.asarray(longitude, float)).reshape(-1, 3)
        k = min(count, self.n_nodes)
        _, index = self.tree.query(points, k=k)
        index = np.asarray(index, dtype=np.int64).reshape(len(points), k)
        cosine = np.clip(np.einsum('pkj,pj->pk', self.xyz[index], points), -1, 1)
        distance = EARTH_RADIUS_M * np.arccos(cosine)
        offset = points[:, None, :] - self.xyz[index]
        local = np.einsum('pkij,pkj->pki', self.frames[index], offset) * EARTH_RADIUS_M / self.node_spacing_m
        return index, distance, local


def torch_buffers(mesh: LatentMesh, device=None, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Constant tensors used inside the model (registered as non-persistent buffers)."""
    dual = mesh.dual
    return {
        'xyz': torch.tensor(mesh.xyz, dtype=dtype, device=device),
        'node_frames': torch.tensor(mesh.frames, dtype=dtype, device=device),
        'senders': torch.tensor(mesh.senders, device=device),
        'receivers': torch.tensor(mesh.receivers, device=device),
        'edge_features': torch.tensor(mesh.edge_features, dtype=dtype, device=device),
        'areas': torch.tensor(mesh.areas_m2, dtype=torch.float64, device=device),
        'dual_pairs': torch.tensor(dual['pairs'], device=device),
        'dual_midpoint': torch.tensor(dual['midpoint'], dtype=dtype, device=device),
        'dual_normal': torch.tensor(dual['normal'], dtype=dtype, device=device),
        'dual_length': torch.tensor(dual['length'], dtype=dtype, device=device),
    }


def _transport(vector: torch.Tensor, origin: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    """Parallel transport of tangent vectors along the shortest geodesic (unit sphere)."""
    return vector - ((vector * destination).sum(-1, keepdim=True)
                     / (1 + (origin * destination).sum(-1, keepdim=True))) * (origin + destination)


def divergence(wind: torch.Tensor, xyz: torch.Tensor, pairs: torch.Tensor, midpoint: torch.Tensor,
               normal: torch.Tensor, length: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Finite-volume horizontal divergence of Cartesian tangent wind [N,...,3] -> [N,...], s^-1.

    Exact Voronoi dual (geometry.py): one midpoint flux per dual edge from
    parallel-transported winds, added with opposite signs.
    """
    left, right = pairs[:, 0], pairs[:, 1]
    extra = wind.dim() - 2
    shape = (-1,) + (1,) * extra + (3,)
    mid = midpoint.view(shape)
    vl = _transport(wind[left], xyz[left].view(shape), mid)
    vr = _transport(wind[right], xyz[right].view(shape), mid)
    flux = (0.5 * (vl + vr) * normal.view(shape)).sum(-1) * length.view((-1,) + (1,) * extra)
    total = torch.zeros(wind.shape[:-1], dtype=flux.dtype, device=flux.device)
    total = total.index_add(0, left, flux).index_add(0, right, -flux)
    return total / areas.to(flux.dtype).view((-1,) + (1,) * extra)


def gradient(scalar: torch.Tensor, xyz: torch.Tensor, pairs: torch.Tensor, normal: torch.Tensor,
             length: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Green-Gauss gradient of a cell scalar [N] -> tangent Cartesian vector [N,3], per metre.

    Uses the closed-contour identity grad s = (1/A) oint (s - s_c) n dl, so a
    constant field has exactly zero gradient. The shared face value is the
    pair mean; both cells therefore receive the same 0.5 (s_r - s_l) n L.
    """
    left, right = pairs[:, 0], pairs[:, 1]
    contribution = (0.5 * (scalar[right] - scalar[left]))[:, None] * normal * length[:, None]
    total = torch.zeros((len(scalar), 3), dtype=contribution.dtype, device=contribution.device)
    total = total.index_add(0, left, contribution).index_add(0, right, contribution)
    vector = total / areas.to(contribution.dtype)[:, None]
    return vector - (vector * xyz).sum(-1, keepdim=True) * xyz
