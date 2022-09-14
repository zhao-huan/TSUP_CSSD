import os
import sys

data_path = sys.argv[1]

segments = f"{data_path}/segments"
utt2spk = {}
spk2utt = {}
for line in open(segments):
    subutt, utt, _, _ = line.strip().split()
    utt2spk[subutt] = utt
    spk2utt.setdefault(utt, []).append(subutt)

with open(f"{data_path}/utt2spk", "w") as f:
    for utt, spk in utt2spk.items():
        print(f"{utt} {spk}", file=f)

with open(f"{data_path}/spk2utt", "w") as f:
    for spk, utts in spk2utt.items():
        print(f"{spk} {' '.join(utt for utt in utts)}", file=f)
