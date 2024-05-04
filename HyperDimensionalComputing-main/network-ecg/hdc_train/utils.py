import argparse
import torch
import torch.nn as nn
import torchhd
from torchhd import embeddings

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 1000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
N_GRAM_SIZE = 4
DOWNSAMPLE = 5
MAX = 5.1560
MIN = -5.2297
MAX_VAL =  52000
MIN_VAL = -53000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/hpc/iwi3/iwi3083h/data/CPSC', help='Directory for input data')
    parser.add_argument('--epoch', type=int, default=1, help='Number of Epochs')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory for output data')
    parser.add_argument('--label-csv', type=str, default='/home/hpc/iwi3/iwi3083h/network-ecg/train/labels/labels.csv', help='Directory for labels.csv')
    parser.add_argument('--classes', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--dimensions', type=int, default=10000, help='Dimension of Hypervectors')
    parser.add_argument('--num-levels', type=int, default=1000, help='Number of quanitized levels for level encoding')
    parser.add_argument('--max-val', type=int, default=52000, help='Maximum signal value for level encoding')
    parser.add_argument('--min-val', type=int, default=-53000, help='Minimum signal value for level encoding')
    parser.add_argument('--use-ngram', action='store_true', help='Flag to use ngram for binding')
    parser.add_argument('--n-gram', type=int, default=4, help='N Gram size for ngrams of signals')
    parser.add_argument('--sampling', type=int, default=6000, help='For eg. 6000 means last 30 sec at 200 Hz.')
    parser.add_argument('--add-online', action='store_true', help='Enables Online Add, Only updates the prototype vectors on wrongly predicted inputs')
    parser.add_argument('--encoding', type=str, default="cv", help='Possible Encoding Values: fv, cv, fChVal, featChVal, chFeatVal')
    return parser.parse_args()


class FeatxValEncoder(nn.Module):
    def __init__(self, out_features, timestamps, channels, ngrams=True):
        super(FeatxValEncoder, self).__init__()
        self.ngrams = ngrams
        self.channels = channels
        self.timestamps = timestamps
        self.out_features = out_features
        self.features = embeddings.Random(timestamps, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features,low=MIN_VAL, high=MAX_VAL)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.reshape(input, (1, self.channels, self.timestamps))
        signal = self.signals(input)
        # signal = signal.resize(1, self.channels, self.timestamps, self.out_features)
        samples = torchhd.bind(signal, self.features.weight.unsqueeze(0))
        samples = torchhd.multiset(samples)
        samples = torchhd.hard_quantize(samples)
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE) if self.ngrams else torchhd.multiset(samples)
        return torchhd.hard_quantize(sample_hv)

class ChxValEncoder(nn.Module):
    def __init__(self, out_features, timestamps, channels, ngrams=True):
        super(ChxValEncoder, self).__init__()
        self.channels = embeddings.Random(channels, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, low=MIN_VAL, high=MAX_VAL)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        samples = torchhd.multiset(samples)
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)
        return torchhd.hard_quantize(sample_hv)


class FChCombxValEncoder(nn.Module):
    def __init__(self, out_features, timestamps, channels, ngrams=True):
        super(FChCombxValEncoder, self).__init__()
        self.ngrams = ngrams
        self.channels = channels
        self.timestamps = timestamps
        self.out_features = out_features
        self.n_feat_ch = timestamps*channels
        self.feat_ch = embeddings.Random(self.n_feat_ch, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features,low=MIN_VAL, high=MAX_VAL)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.reshape(input, (1,self.n_feat_ch))
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.feat_ch.weight.unsqueeze(0))
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE) if self.ngrams else torchhd.multiset(samples)
        return torchhd.hard_quantize(sample_hv)
    
class FeatxChxValEncoder(nn.Module):
    def __init__(self, out_features, timestamps, channels, ngrams=True):
        super(FeatxChxValEncoder, self).__init__()
        self.ngrams = ngrams
        self.ch = channels
        self.timestamps = timestamps
        self.out_features = out_features
        self.features = embeddings.Random(timestamps, out_features)
        self.channels = embeddings.Random(channels, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, low=MIN_VAL, high=MAX_VAL)  
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        samples = torchhd.multiset(samples)
        samples = torchhd.hard_quantize(samples)
        samples = torchhd.bind(samples, self.features.weight.unsqueeze(0))
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE) if self.ngrams else torchhd.multiset(samples)
        return torchhd.hard_quantize(sample_hv)
    
class ChxFeatxValEncoder(nn.Module):
    def __init__(self, out_features, timestamps, channels, ngrams=True):
        super(ChxFeatxValEncoder, self).__init__()
        self.ch = channels
        self.ngrams = ngrams
        self.timestamps = timestamps
        self.out_features = out_features
        self.features = embeddings.Random(timestamps, out_features)
        self.channels = embeddings.Random(channels, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, low=MIN_VAL, high=MAX_VAL)  
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.reshape(input, (1, self.ch, self.timestamps))
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.features.weight.unsqueeze(0))
        samples = torchhd.multiset(samples)
        samples = torchhd.hard_quantize(samples)
        samples = torchhd.bind(samples, self.channels.weight.unsqueeze(0))
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE) if self.ngrams else torchhd.multiset(samples)
        return torchhd.hard_quantize(sample_hv)
    


