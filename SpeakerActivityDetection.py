import os
import random
import argparse
import kaldiio
import numpy as np
import soundfile as sf
import tqdm
import torch
import torch.nn as nn
from torch.nn import DataParallel as DP
import traceback
from sad import get_SAD

from tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--seed', type=int, default=1986)
parser.add_argument('--threshold', type=float, default=0.55)
parser.add_argument('--chunk-size', type=float, default=16)
parser.add_argument('--chunk-shift', type=float, default=4)

args = parser.parse_args()


def generate_chunks(wav, lengths=16.0, shift=4.0):
    lengths = int(lengths * 16000)
    shift = int(shift * 16000)
    results = []
    wav_len = wav.shape[0]
    start = 0
    while start < wav_len - (lengths - shift):
        results.append(wav[start:min(start+lengths, wav.shape[0])])
        start += shift
    return results

def post_process(labels, args):
    total_frame = 200 + (len(labels) - 2) * 50 + labels[-1][150:].shape[0]
    sum_labels = np.zeros((total_frame, 1))
    last_labels = np.zeros((total_frame, 1))
    repeat_num = np.zeros((total_frame, 1))
    start = 0
    for label in labels:
        lengths = label.shape[0]
        sum_labels[start:min(start+lengths, total_frame)] += label
        repeat_num[start:min(start+lengths, total_frame)] += 1
        start += 50
    assert start - 50 + lengths == total_frame, f"{start} {lengths} {total_frame}"
    sum_labels = sum_labels / repeat_num

    for i in range(last_labels.shape[0]):
        l = max(0, i - 5)
        r = i + 5
        last_labels[i] = np.median(sum_labels[l:r], axis=0)

    return last_labels


def make_segments(rttm, save_file_name):
    with open(save_file_name, 'w') as f:
        for sess, segs in rttm.items():
            segs = sorted(segs, key=lambda e:e[0])
            for idx, (start, end, spk) in enumerate(segs):
                if end - start <= 0:
                    continue
                line = f"{sess}-{idx+1:04} {sess} {start} {end}"
                print(line, file=f)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    args.device = torch.device("cuda", 0)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set torch deterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    compute_features = FeatureWrapper(
        feature_module=Fbank(
            sample_rate=16000,
            snip_edges=False
        ),
        feature_normalization=SentenceFeatureNormalization(
            mean_norm=True,
            std_norm=True
        ),
        expands_dim=-1
    )

    model = get_SAD().to(args.device)
    model_state_dict = torch.load(args.model_path, map_location=args.device)['model']
    model_state_dict = {k.replace("module.", ""):v for k,v in model_state_dict.items()}
    #model = DP(model)
    model.load_state_dict(model_state_dict)
    print("load model finished")

    model.eval()

    wav_scp = read_wav_scp(os.path.join(args.data_path, "wav.scp"))

    rttm = {}
    with torch.no_grad():
        for session, path in wav_scp.items():
            print("processing session {}".format(session))
            wav, sr = sf.read(wav_scp[session])
            total_duration = wav.shape[0] / sr

            chunks = generate_chunks(wav, args.chunk_size, args.chunk_shift)
            labels = []

            for idx, chunk in tqdm.tqdm(enumerate(chunks)):
                chunk = torch.from_numpy(chunk).reshape(1, -1).float().to(args.device)
                feats, lens = compute_features(chunk)
                predictions = model(feats)
                predictions = torch.sigmoid(predictions)
                predictions = predictions.detach().cpu().squeeze(0)
                
                labels.append(predictions.numpy())

            labels = post_process(labels, args)
            labels = np.where(labels < args.threshold, 0, 1)

            num_spks = 1
            spk_start = [0 for _ in range(num_spks)]
            spk_ju = [0 for _ in range(num_spks)]
            for i in range(labels.shape[0]):
                for j in range(num_spks):
                    if labels[i, j] == 1:
                        if spk_ju[j] == 0:
                            spk_start[j] = i
                            spk_ju[j] = 1
                    else:
                        if spk_ju[j] == 1:
                            time_start = (spk_start[j]) / 12.5
                            time_end = min((i - 1) / 12.5, total_duration)
                            if time_end - time_start > 0.2:
                                rttm.setdefault(session, []).append((time_start, time_end, j))
                            spk_ju[j] = 0

            for j in range(num_spks):
                if spk_ju[j] == 1:
                    time_start = (spk_start[j]) / 12.5
                    time_end = min((i - 1) / 12.5, total_duration)
                    if time_end - time_start > 0.2:
                        rttm.setdefault(session, []).append((time_start, time_end, j))
                    spk_ju[j] = 0

    make_segments(rttm, args.save_path)

