"""SHD, SSC and N-MNIST through tonic (events binned into 4 ms frames), and Google Speech
Commands v0.02 as 40-bin log-mel filterbank frames at a 4 ms shift. Frames are cached on disk on
first touch; scripts/build_cache.py fills the cache in parallel."""
import os

import h5py
import numpy as np
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Subset

DATASETS = ('SHD', 'SSC', 'NMNIST', 'GSC')
CLIP_MS = {'SHD': 700, 'SSC': 1000, 'NMNIST': 310, 'GSC': 1000}
INPUT_BIN = {'SHD': 5, 'SSC': 5, 'NMNIST': 5, 'GSC': 1}
CHANNEL_SHIFT = {'SHD': 40, 'SSC': 40, 'NMNIST': 40, 'GSC': 2}
SHD_SEEN_SPEAKERS = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11}


class GSCFbank(torch.utils.data.Dataset):
    """Speech Commands v0.02, 35 classes, Kaldi filterbank with 40 mel bins and a 25 ms window."""

    def __init__(self, data_dir, subset, frame_shift_ms):
        import torchaudio
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(os.path.join(data_dir, 'SpeechCommands'),
                                                    url='speech_commands_v0.02', subset=subset, download=False)
        root = self.ds._path
        self.labels = sorted(d for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d)) and not d.startswith('_'))
        assert len(self.labels) == 35, self.labels
        self.shift = float(frame_shift_ms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import torchaudio
        wav, sr, label = self.ds[i][:3]
        x = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=40, frame_length=25.0,
                                              frame_shift=self.shift, sample_frequency=sr)
        return x.numpy().astype('float32'), self.labels.index(label)


def gsc_standardizer(stats_path, device):
    """Per-bin standardization of the log-mel input with training-set statistics."""
    st = np.load(stats_path)
    mean = torch.tensor(st['mean'], device=device)
    std = torch.tensor(st['std'], device=device)
    return lambda x: (x - mean) / std


def get_splits(dataset, data_dir, time_window_us):
    if dataset == 'GSC':
        shift = time_window_us / 1000.0
        return (GSCFbank(data_dir, 'training', shift), GSCFbank(data_dir, 'testing', shift), 35, 40)
    cls = getattr(tonic.datasets, dataset)
    sensor = cls.sensor_size
    transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor, time_window=time_window_us)])
    if dataset == 'SSC':
        train = cls(save_to=data_dir, transform=transform, split='train')
        test = cls(save_to=data_dir, transform=transform, split='test')
    else:
        train = cls(save_to=data_dir, transform=transform, train=True)
        test = cls(save_to=data_dir, transform=transform, train=False)
    num_classes = {'SHD': 20, 'SSC': 35, 'NMNIST': 10}[dataset]
    return train, test, num_classes, int(np.prod(sensor))


def load_data(dataset, batch_size, time_window_us=4000, data_dir='./data', cache_dir='./cache'):
    """Returns (train_loader, test_loader, num_classes, input_width). The test set is visited in
    one fixed random order (SSC is stored sorted by class)."""
    assert dataset in DATASETS, dataset
    train_raw, test_raw, num_classes, width = get_splits(dataset, data_dir, time_window_us)
    key = f'{time_window_us}us'
    train = tonic.DiskCachedDataset(train_raw, cache_path=os.path.join(cache_dir, dataset, f'train_{key}'))
    test = tonic.DiskCachedDataset(test_raw, cache_path=os.path.join(cache_dir, dataset, f'test_{key}'))
    collate = tonic.collation.PadTensors(batch_first=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    perm = torch.randperm(len(test), generator=torch.Generator().manual_seed(12345)).tolist()
    test_loader = DataLoader(Subset(test, perm), batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader.perm = perm
    return train_loader, test_loader, num_classes, width


def shd_unseen_mask(data_dir, perm):
    """Mask over the permuted SHD test set: samples from speakers absent from the training set."""
    with h5py.File(os.path.join(data_dir, 'SHD', 'shd_test.h5'), 'r') as h:
        speaker = np.array(h['extra']['speaker'])
    unseen = np.array([s not in SHD_SEEN_SPEAKERS for s in speaker])
    return unseen[np.array(perm)]
