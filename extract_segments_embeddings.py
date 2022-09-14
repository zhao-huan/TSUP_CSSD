import os
import torch
import tqdm
import argparse
import kaldiio
import numpy as np
import traceback
from sklearn.preprocessing import normalize

import soundfile as sf
from resnet import create_ResNet34Tiny
from tools import *

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)

args = parser.parse_args()

def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split())
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns

class AudioReadHolder(object):
    def __init__(self, sample_rate):
        self.sr = sample_rate

    def __call__(self, wav_path, start=0, end=None):
        wav_path = wav_path.strip()

        if wav_path.endswith("|"):
            wav_path = wav_path[:-1]
            utt = read_pipe_wav(wav_path, sample_rate=self.sr, start=start, end=end)
        else:
            utt, sr = sf.read(wav_path, start=start, stop=end)
            assert self.sr == sr, f"Sample rate {self.sr} vs {sr} is different!"
        # if len(utt.shape) > 1:
        #     utt = utt.sum(axis=1)
        return utt.astype(np.float32)

class SpeakerVerificationEvaluateProcessor:
    def __init__(self, args):
        self.args = args

        opts_default = {
            "checkpoint": None,
            "seed": 1986,
            "device": "cuda:0",
        }

        for arg, default in opts_default.items():
            setattr(self, arg, getattr(self.args, arg, default))

        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))
        torch.manual_seed(self.seed)

        self.compute_features = FeatureWrapper(
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
        self.embedding_model = create_ResNet34Tiny()

        self.modules = torch.nn.ModuleDict({
            "compute_features": self.compute_features,
            "embedding_model": self.embedding_model
        }).to(self.device)
        self.checkpoint = args.model_path

        self.wav_scp = read_wav_scp(os.path.join(self.args.data_path, "wav.scp"))
        self.segments = read_segments(os.path.join(self.args.data_path, "segments"))

        save_str = f"ark,scp:{os.path.join(self.args.save_path, 'embeddings.ark')},{os.path.join(self.args.save_path, 'embeddings.scp')}"
        self.writer = kaldiio.WriteHelper(save_str)
        self._load_and_init()

    def _load_and_init(self):
        self.modules.to(self.device)
        print(f"Load checkpoint from {self.checkpoint}")

        checkpoint = torch.load(self.checkpoint)

        for name, module in self.modules.items():
            if any(p.requires_grad for p in module.parameters()):
                try:
                    state_dict = checkpoint[name]
                    if hasattr(module, 'module'):
                        self.modules[name].module.load_state_dict(state_dict)
                    else:
                        self.modules[name].load_state_dict(state_dict)
                except:
                    print(f"load {name}'s parameters failed")

    def compute_embedding(self):
        self.modules.eval()

        #wav_rows = list(zip(*load_n_col(self.config.data_path)))
        reader = AudioReadHolder(16000)



        with torch.no_grad():
            for index, (session, segs) in tqdm.tqdm(enumerate(self.segments.items())):
                for utt_id, start, end in segs:
                    try:
                        start = int(start * 16000)
                        end = int(end * 16000)
                        wavs = reader(self.wav_scp[session], start=start, end=end)
                        if len(wavs.shape) == 2:
                            wavs = torch.from_numpy(wavs[:, 0].copy()).unsqueeze(0)
                        else:
                            wavs = torch.from_numpy(wavs.copy()).unsqueeze(0)

                        wavs = wavs.to(self.device)

                        lens = torch.ones(wavs.shape[0]).to(self.device)
                        lens *= wavs.shape[-1]
                        lens = lens.long()

                        feats, lens = self.modules["compute_features"](wavs, lens)
                        embeddings = self.modules["embedding_model"](feats, lens)

                        embeddings = embeddings.detach().cpu().numpy()
                        #embeddings = normalize(embeddings, norm='l2').flatten()
                        embeddings = embeddings.flatten()
                        self.writer(utt_id, embeddings)
                    except:
                        with open('error', 'a+') as f:
                            print(f"{index} {utt_id} {wavs.shape}", file=f)
                            print(traceback.format_exc(), file=f)

        self.writer.close()

if __name__ == "__main__":

    processor = SpeakerVerificationEvaluateProcessor(args)
    processor.compute_embedding()
