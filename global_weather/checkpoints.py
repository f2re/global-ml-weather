"""Schema-first checkpoint loading: do not partially load incompatible weights."""
from pathlib import Path
import torch


def save_checkpoint(path, model):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix+'.tmp')
    torch.save({'schema': model.get_extra_state(), 'state_dict': model.state_dict()}, tmp)
    tmp.replace(target)


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=model.xyz.device, weights_only=True)
    if checkpoint['schema'] != model.get_extra_state():
        raise ValueError('Incompatible checkpoint; model was not modified.')
    expected = model.state_dict()
    supplied = checkpoint['state_dict']
    if supplied.keys() != expected.keys():
        raise ValueError('Checkpoint keys mismatch; model was not modified.')
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            if not isinstance(supplied[key], torch.Tensor) or supplied[key].shape != expected[key].shape:
                raise ValueError('Checkpoint shapes mismatch; model was not modified.')
    for key, reference in model.named_buffers():
        value = supplied[key]
        if reference.is_sparse:
            a,b=reference.coalesce(),value.coalesce()
            equal=torch.equal(a.indices(),b.indices()) and torch.equal(a.values(),b.values())
        else:
            equal=torch.equal(value,reference)
        if not equal:
            raise ValueError('Immutable grid/normalisation buffers differ; model was not modified.')
    for value in supplied.values():
        if isinstance(value,torch.Tensor):
            values=value.coalesce().values() if value.is_sparse else value
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite checkpoint tensor; model was not modified.')
    if supplied.get('_extra_state') != checkpoint['schema']:
        raise ValueError('Checkpoint internal schema mismatch; model was not modified.')
    model.load_state_dict(supplied, strict=True)
