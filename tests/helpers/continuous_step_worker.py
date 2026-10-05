"""Real subprocess crash harness. All values are synthetic persistence tests."""
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from global_weather.continuous.store import CampaignStore
from global_weather.continuous.checkpointing import StepTrainer


def setup(root):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    random.seed(18)
    np.random.seed(19)
    store = CampaignStore(root)
    store.add_range('2000-02-01', '2000-02-28')
    uses = []
    for day in (10, 11, 12):
        sample = store.register_sample({'issue_time': f'2000-02-{day:02d}T12:00:00Z',
            'inputs': [{'provider': 'synthetic', 'object_id': f'test/{day}', 'revision': '1', 'sha256': 'a'*64}],
            'targets': [{'provider': 'synthetic', 'object_id': f'target/{day}', 'revision': '1', 'sha256': 'b'*64}],
            'modalities': ['station'], 'transform_sha256': 'c'*64})
        uses.append(store.plan_use(sample['id'])['id'])
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.95)
    return store, model, optimizer, scheduler, uses


def operation(model, optimizer):
    def run():
        model.train()
        x = torch.rand(2, 3) + random.random() + float(np.random.random())
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return {'loss': float(loss.detach())}
    return run


def main():
    root, point = Path(sys.argv[1]), sys.argv[2]
    store, model, optimizer, scheduler, uses = setup(root)
    armed = False
    def fail(where):
        if armed and point == where:
            os._exit(77)
    original_save = torch.save
    def save(obj, stream, *args, **kwargs):
        if armed and point == 'truncated_write':
            stream.write(b'INCOMPLETE STATE')
            stream.flush()
            os._exit(77)
        return original_save(obj, stream, *args, **kwargs)
    torch.save = save
    with StepTrainer(store, model, optimizer, identity={'data_kind': 'synthetic', 'purpose': 'crash_test'},
                     scheduler=scheduler, failpoint=fail) as trainer:
        armed = True
        for index, use in enumerate(uses):
            trainer.step([use], operation(model, optimizer), cursor={'block': 'synthetic', 'next_index': index+1})


if __name__ == '__main__':
    main()
