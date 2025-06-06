#!/bin/bash

# =======================
# Parameters (Change these if needed)
# =======================
DATA_PATH="/home/users/ntu/adnan002/scratch/data/DIHARD3_vbx_vad_osd/third_dihard_challenge_eval/data"  # Path to the wav directory
MAX_SPEAKER=8 # Maximum number of speakers
SOURCE_MODEL="../pretrained_models/ecapa-tdnn.model" # Path to the source model
MODEL_PATH="pretrained_models/model_0015_newlongsimdata_pretrain.model" # Path to the model for TS-VAD
OUTPUT_PATH="exps/infer_vad_osd_d8"  # Output path
TEST_SHIFT=4  # Test shift value (Use: 0.5 for slightly lower DER; slower)
N_CPU=12  # Number of CPU cores
HF_TOKEN="your_hugging_face_token" # (Request access for https://huggingface.co/pyannote/segmentation-3.0)
# Optional: Ground truth RTTM file (leave empty if not available)
GROUNDTRUTH_RTTM="/home/users/ntu/adnan002/scratch/data/DIHARD3_vbx_vad_osd/third_dihard_challenge_eval/data/all.rttm"

# =======================
# Stages (Change these if needed)
# =======================
START_STAGE=2  # Set start stage (1, 2, 3, 4)
END_STAGE=3    # Set end stage (1, 2, 3, 4)

# Set up the output path with a timestamp
TIME=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PATH="${OUTPUT_PATH}_${TIME}"

# =======================
# Stage 1: VBx Prediction
# =======================
if [ $START_STAGE -le 1 ] && [ $END_STAGE -ge 1 ]; then
    echo "=============================="
    echo "Running VBx prediction (Stage 1)..."
    echo "=============================="
    cd VBx
    python predict.py --in-wav-dir "$DATA_PATH" --hf-token "$HF_TOKEN"
    if [ $? -ne 0 ]; then
        echo "Error: VBx prediction failed."
        exit 1
    fi
    cd ..

    # If ground truth RTTM is provided, perform scoring after VBx clustering
    if [ ! -z "$GROUNDTRUTH_RTTM" ]; then
        echo "Performing scoring after VBx clustering..."

        # Combine all RTTM files into a single computed_rttm file
        cat "${DATA_PATH}/rttm/"*.rttm > "${DATA_PATH}/computed_rttm_vbx.rttm"

        # Run scoring for VBx clustering
        cd ts-vad/tools/SCTK-2.4.12/src/md-eval
        ./md-eval.pl -c 0.25 -s "${DATA_PATH}/computed_rttm_vbx.rttm" -r "$GROUNDTRUTH_RTTM"
        ./md-eval.pl -c 0.00 -s "${DATA_PATH}/computed_rttm_vbx.rttm" -r "$GROUNDTRUTH_RTTM"
        cd ../../../../../

        echo "VBx scoring completed."
        echo "VBx computed RTTM file: ${DATA_PATH}/computed_rttm_vbx.rttm"
    fi

    echo "Stage 1 completed."
fi

# =======================
# Stage 2: Prepare Embeddings
# =======================
if [ $START_STAGE -le 2 ] && [ $END_STAGE -ge 2 ]; then
    echo "=============================="
    echo "Preparing embeddings (Stage 2)..."
    echo "=============================="
    cd ts-vad/prepare
    python prepare_embeddings.py \
        --data_path "$DATA_PATH" \
        --max_speaker "$MAX_SPEAKER" \
        --source "$SOURCE_MODEL"
    if [ $? -ne 0 ]; then
        echo "Error: Embedding preparation failed."
        exit 1
    fi
    cd ../..
    echo "Stage 2 completed."
fi

# =======================
# Stage 3: TS-VAD Evaluation
# =======================
if [ $START_STAGE -le 3 ] && [ $END_STAGE -ge 3 ]; then
    echo "=============================="
    echo "Running TS-VAD evaluation (Stage 3)..."
    echo "=============================="
    cd ts-vad
    # ignore train_list and train_path (we only do evaluation). it will not be used when --eval is set
    python main.py \
        --train_list "${DATA_PATH}/ts_infer.json" \
        --eval_list "${DATA_PATH}/ts_infer.json" \
        --train_path "$DATA_PATH" \
        --eval_path "$DATA_PATH" \
        --save_path "$OUTPUT_PATH" \
        --rs_len 4 \
        --test_shift "$TEST_SHIFT" \
        --min_silence 0.32 \
        --min_speech 0.00 \
        --threshold 0.50 \
        --n_cpu "$N_CPU" \
        --eval \
        --init_model "$MODEL_PATH" \
        --max_speaker "$MAX_SPEAKER"
    
    echo "TS-VAD evaluation completed. The res_rttm file is located at: $OUTPUT_PATH/res_rttm"
    
    if [ $? -ne 0 ]; then
        echo "Error: TS-VAD evaluation failed."
        exit 1
    fi

    # If ground truth RTTM is provided, perform scoring after TS-VAD
    if [ ! -z "$GROUNDTRUTH_RTTM" ]; then
        echo "Performing scoring after TS-VAD evaluation..."

        # Use the res_rttm generated by TS-VAD as computed RTTM
        COMPUTED_RTTM="${OUTPUT_PATH}/res_rttm"
        COMPUTED_RTTM=$(pwd)/$COMPUTED_RTTM

        # Run scoring for TS-VAD
        cd tools/SCTK-2.4.12/src/md-eval
        ./md-eval.pl -c 0.25 -s "$COMPUTED_RTTM" -r "$GROUNDTRUTH_RTTM"
        ./md-eval.pl -c 0.00 -s "$COMPUTED_RTTM" -r "$GROUNDTRUTH_RTTM"
        cd ../../../../../

        echo "TS-VAD scoring completed."
    fi

    echo "Stage 3 completed."
fi

# =======================
# Stage 4: Postprocessing VAD
# =======================
if [ $START_STAGE -le 4 ] && [ $END_STAGE -ge 4 ]; then
    echo "=============================="
    echo "Running Postprocessing VAD (Stage 4)..."
    echo "=============================="

    # Define paths for postprocessing
    POSTPROCESS_RTTM="${OUTPUT_PATH}/res_rttm_postprocessing.rttm"

    # Run postprocessing_VAD.py
    python postprocessing_VAD.py \
        --input_path "${DATA_PATH}/wav" \
        --input_rttm "${OUTPUT_PATH}/res_rttm" \
        --output_dir "$OUTPUT_PATH" \
        --pyannote_segementation_token "$HF_TOKEN" \
        --output_rttm_file "$POSTPROCESS_RTTM"

    if [ $? -ne 0 ]; then
        echo "Error: Postprocessing VAD failed."
        exit 1
    fi

    echo "Postprocessing VAD completed."
    echo "Postprocessed RTTM file: $POSTPROCESS_RTTM"

    # Perform scoring on the postprocessed RTTM
    if [ ! -z "$GROUNDTRUTH_RTTM" ]; then
        echo "Performing scoring on postprocessed RTTM..."

        # set POSTPROCESS_RTTM to full path
        POSTPROCESS_RTTM=$(pwd)/$POSTPROCESS_RTTM

        cd ts-vad/tools/SCTK-2.4.12/src/md-eval
        ./md-eval.pl -c 0.25 -s "$POSTPROCESS_RTTM" -r "$GROUNDTRUTH_RTTM"
        ./md-eval.pl -c 0.00 -s "$POSTPROCESS_RTTM" -r "$GROUNDTRUTH_RTTM"
        cd ../../../../../

        echo "Scoring on postprocessed RTTM completed."
    fi

    echo "Stage 4 completed."
fi

echo "=============================="
echo "Script finished successfully."
echo "=============================="
