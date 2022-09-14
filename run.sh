
stage=1

if [ $stage -le 1 ]; then
    echo "preparing data"
    magicdata_path=$1

    mkdir -p exp/cssd

    ls $magicdata_path/*.wav > exp/cssd/wav_list
    cat exp/cssd/wav_list | awk -v FS='/' '{print $7}' | awk -v FS='.' '{print $1}' > exp/cssd/session_list
    paste exp/cssd/session_list exp/cssd/wav_list > exp/cssd/wav.scp
fi

if [ $stage -le 2 ]; then
    echo "do SAD"
    python SpeakerActivityDetection.py --data-path exp/cssd \
        --model-path models/sad.pth \
        --save-path exp/cssd/segments
    python fix_segments_folder.py exp/cssd
fi

if [ $stage -le 3 ]; then
    echo "extract embeddings"

    utils/data/get_uniform_subsegments.py --max-segment-duration=16.0 --overlap-duration=8.0 \
        --max-remaining-duration=8.0 --constant-duration=True \
        exp/cssd/segments > exp/cssd/subsegments
    utils/data/subsegment_data_dir.sh exp/cssd exp/cssd/subsegments exp/cssd/subdata

    python extract_segments_embeddings.py --model-path models/resnet_finetune.pth.tar \
        --data-path exp/cssd/subdata \
        --save-path exp/cssd

fi

if [ $stage -le 4 ]; then
    echo "do clustering"
    python cluster.py --out-rttm-dir exp/cssd/out_rttm \
        --ark-file exp/cssd/embeddings.ark \
        --segments-file exp/cssd/subdata/segments \
        --threshold 0.999 \
        --spk-num 2

    cat exp/cssd/out_rttm/*.rttm > exp/cssd/cssd.rttm
fi
