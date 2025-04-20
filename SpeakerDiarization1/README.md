# End to End Diarization Pipeline

This guide will walk you through setting up VBx and TSVAD_pytorch environment to run end-to-end diarization pipeline. This guide focuses on evaluation (not training).

## 1. Conda Installation

```bash
# Ignore if you already have conda setup
cd /workspace
rm -rf miniconda3/

INSTALLER="./Miniconda3-latest-Linux-x86_64.sh"

if [ ! -f "$INSTALLER" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
else
    echo "Installer $INSTALLER already exists."
fi

INSTALL_DIR="/workspace/miniconda3"

bash "$INSTALLER" -b -p "$INSTALL_DIR"
export PATH="/workspace/miniconda3/bin:$PATH"
```

## 2. Pull from GitHub

```bash
git clone https://github.com/adnan-azmat/TSVAD_pytorch.git
cd TSVAD_pytorch

conda env create --name wespeak2 --file=wespeak2.yml
source activate wespeak2

cd ts-vad
git checkout u/adnan/e2epipeline
mkdir pretrained_models
cd pretrained_models

gdown 1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb # [WavLM-Base+.pt](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link)
gdown 1E-ju12Jy1fID2l4x-nj0zB5XUHSKWsRB # [ecapa-tdnn.model](https://drive.google.com/file/d/1E-ju12Jy1fID2l4x-nj0zB5XUHSKWsRB/view?usp=drive_link)
```

## 3. Evaluation

Before running the evaluation script, ensure you have the following:

1. **List of `.wav` Files**: A collection of audio recordings in `.wav` format that you wish to evaluate.
2. **Ground Truth RTTM File (Optional)**: If available, provide an RTTM file containing ground truth speaker rttm. This file will be used for calculating DER

Execute the provided Bash script: `script.sh` to start the evaluation process

Before running the script, you can adjust various parameters to suit your setup:

- **DATA_PATH**: Path to the directory containing `wav` folder (path to `data` folder in above example)
- **MAX_SPEAKER**: Maximum number of speakers to handle / maximum number of speakers the TS-VAD model can handle.
- **MODEL_PATH**: Path to the pre-trained model for TS-VAD.
- **TEST_SHIFT**: Test shift value for TS-VAD. Use 0.5 for better DER (slower).
- **N_CPU**: Number of CPU cores to utilize.
- **GROUNDTRUTH_RTTM**: Path to the ground truth RTTM file (optional).
- **START_STAGE & END_STAGE**: Define which stages to run.
- **HF_TOKEN**: HuggingFace Token

Note: 
- Request access for https://huggingface.co/pyannote/segmentation-3.0. Paste HuggingFace token in HF_TOKEN param
- For model path you can use a pre-trained TS-VAD model from here: https://drive.google.com/file/d/1YBW_XNck6hkJ8ouXKH2aR_9pHhZtWWwN/view?usp=sharing
- The script creates a folder called 'rttm' in `DATA_PATH`. If it already exists consider renaming it 
-----
## 4. Progression through the stages

This section is just for insight. `script.sh` handles all the stages

The `script.sh` is divided into three main stages, each responsible for a specific part of the evaluation pipeline.

Organize your data directory to include the necessary subdirectories for audio files and RTTM annotations. The script will generate additional directories and files as it progresses through each stage.

Initial setup:

```
data/
├── wav/
│   ├── file1.wav
│   ├── file2.wav
│   ├── file3.wav
│   └── ...
```

After stage 1 the directory should look like:

```
data/
├── wav/
│   ├── file1.wav
│   ├── file2.wav
│   ├── file3.wav
│   └── ...
└── rttm/
    ├── file1.rttm
    ├── file2.rttm
    ├── file3.rttm
    └── ...
```
The files in rttm folder are generated using VBx clustering

After stage 2 the directory should look like:

```
data/
├── wav/
│   ├── file1.wav
│   ├── file2.wav
│   ├── file3.wav
│   └── ...
├── rttm/
│   ├── file1.rttm
│   ├── file2.rttm
│   ├── file3.rttm
│   └── ...
├── target_audio/
│   ├── file1
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   ├── 3.wav
│   │   ├── 4.wav
│   │   └── all.wav
│   ├── file2
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   ├── 3.wav
│   │   ├── 4.wav
│   │   └── all.wav
│   └── ...
├── target_embedding/
│   ├── file1
│   │   ├── 1.pt
│   │   ├── 2.pt
│   │   ├── 3.pt
│   │   └── 4.pt
│   ├── file2
│   │   ├── 1.pt
│   │   ├── 2.pt
│   │   ├── 3.pt
│   │   └── 4.pt
│   └── ...
└── ts_infer.json
```

Stage 3 runs TS-VAD.

If you provide a groundtruth rttm, DER will be calculated for the results from the VBx clustering and TS-VAD models.
