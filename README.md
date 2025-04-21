## 1. Speaker Diarization
To run speaker diarization model, please checkout the `speaker_diarization/` and follow the README in that branch.

## Partial Spoof Detection with Speaker Diarization output 
Inference the pretrained model with single wav file and rttm file:
```
python inference_w_rttm.py --data_path path/to/wav_file --rttm_path path/to/rttm_file --ckpt_path path/to/model_ckpt
```
