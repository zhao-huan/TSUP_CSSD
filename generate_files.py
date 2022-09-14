import os
import argparse
import traceback
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--sets', type=str, nargs='+', default=["train"])
    parser.add_argument('--save-path', type=str, required=True)
    args = parser.parse_args()

    args.data_path = os.path.abspath(args.data_path)
    args.save_path = os.path.abspath(args.save_path)
    wav_list = []
    for s in args.sets:
        for line in open(os.path.join(args.data_path, "DataPartition", s+".tsv")):
            try:
                wav, length = line.strip().split()
            except:
                continue
            wav_list.append(wav)

    session_list = [wav.split(".")[0] for wav in wav_list]

    os.makedirs(args.save_path, exist_ok=True)
    f1 = open(os.path.join(args.save_path, "wav.scp"), "w")
    f2 = open(os.path.join(args.save_path, "utt2spk"), "w")
    f3 = open(os.path.join(args.save_path, "spk2utt"), "w")

    for session in session_list:
        wav_path = os.path.join(args.data_path, "MDT2021S003", "WAV", session+".wav")

        print(f"{session} {wav_path}", file=f1)
        print(f"{session} {session}", file=f2)
        print(f"{session} {session}", file=f3)

    f1.close()
    f2.close()
    f3.close()
