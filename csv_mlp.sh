#!/bin/bash

for aur in 0 1; do
for auc in 0 1; do
for pose in 0 1; do
for gaze in 0 1; do

    # Skip empty feature set
    if [[ $aur -eq 0 && $auc -eq 0 && $pose -eq 0 && $gaze -eq 0 ]]; then
        continue
    fi

    feature_args=""

    [[ $aur  -eq 1 ]] && feature_args="$feature_args --aur"
    [[ $auc  -eq 1 ]] && feature_args="$feature_args --auc"
    [[ $pose -eq 1 ]] && feature_args="$feature_args --pose"
    [[ $gaze -eq 1 ]] && feature_args="$feature_args --gaze"

    for n_frames in 30; do
    for conf_th in 0.8; do
    for n_freq in -1; do
        python csv_mlp.py $feature_args --n-frames $n_frames --conf-th $conf_th --n-freq $n_freq
    done
    done
    done

done
done
done
done