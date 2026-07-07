# REU/
# │
# ├── raw/
# │   ├── 1001_DFA_ANG_XX.flv
# │   └── ...
# ├── audio/
# │   ├── 1001_DFA_ANG_XX.wav
# │   └── ...
# │
# └── openface/
#     ├── 1001_DFA_ANG_XX.csv
#     ├── 1001_DFA_ANG_XX_aligned/
#     ├── 1001_DFA_ANG_XX.hog
#     ├── ...



mkdir -p audio

for f in raw/*.flv; do
    filename=$(basename "$f" .flv)

    ffmpeg -y \
        -i "$f" \
        -vn \
        -ac 1 \
        -ar 16000 \
        -acodec pcm_s16le \
        "audio/${filename}.wav"
done