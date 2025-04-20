#!/usr/bin/env python

import argparse
import os
from datetime import datetime
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

def write_wav_scp(folder_path, output_file):
    with open(output_file, 'w') as f:
        # Loop through all files in the directory
        for filename in os.listdir(folder_path):
            # Check if the file is a .wav or .flac file
            if filename.endswith(".flac") or filename.endswith(".wav"):
                # Get the full path of the file
                file_path = os.path.join(folder_path, filename)
                
                # Write the file name without extension and its path
                file_name = os.path.splitext(filename)[0]
                f.write(f"{file_name} {file_path}\n")

def pyannote_vad(token, wav_scp, save_path):
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0", 
        use_auth_token=token)
    assert os.path.exists(wav_scp), f"wavfile Path does not exist: {wav_scp}"
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    with open(save_path, 'w') as f:
        for line in open(wav_scp, 'r'):
            utt, wav_path = line.strip().split()
            vad = pipeline(wav_path)
            for speech_turn in vad.get_timeline():
                start_time = speech_turn.start
                end_time = speech_turn.end

                # Format: segment_id filename start_time end_time
                line = f"{utt}-{int(start_time*1000):07d}-{int(end_time*1000):07d} {utt} {start_time:.3f} {end_time:.3f}\n"
                f.write(line)
    f.close()
    assert os.path.exists(save_path), f"Lab File processing didn't complete: {save_path}"

def read_rttm(rttm_file):
    rttm_list = []
    
    with open(rttm_file, 'r') as file:
        for line in file:
            # Split the line by spaces
            parts = line.strip().split()
            
            # Each line is expected to have 10 fields, so map them accordingly
            if len(parts) == 10:
                rttm_entry = {
                    'type': parts[0],           # "SPEAKER" type
                    'filename': parts[1],       # File name
                    'channel': int(parts[2]),   # Channel number (usually 1 or 0)
                    'start_time': float(parts[3]),  # Start time of the segment
                    'duration': float(parts[4]),    # Duration of the segment
                    'speaker_id': parts[7]      # Speaker ID
                }
                
                # Append the dictionary entry to the list
                rttm_list.append(rttm_entry)
    
    return rttm_list

def read_vad_segments(vad_file):
    vad_list = []
    
    with open(vad_file, 'r') as file:
        for line in file:
            # Split each line by whitespace
            parts = line.strip().split()
            
            # Each line should contain four elements: segment_id, filename, start_time, and end_time
            if len(parts) == 4:
                vad_entry = {
                    'segment_id': parts[0],         # Segment identifier
                    'filename': parts[1],           # File name
                    'start_time': float(parts[2]),  # Start time of the speech segment
                    'end_time': float(parts[3])     # End time of the speech segment
                }
                
                # Append the dictionary entry to the list
                vad_list.append(vad_entry)
    
    return vad_list

def save_to_rttm(combined_results, output_file):
    with open(output_file, 'w') as file:
        for result in combined_results:
            # RTTM format: SPEAKER <filename> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
            line = f"SPEAKER {result['filename']} {result['channel']} {result['start_time']:.3f} {result['duration']:.3f} <NA> <NA> {result['speaker_id']} <NA> <NA>\n"
            file.write(line)

def combine_vad_rttm(vad_seg_file, rttm_file, output_rttm_file):
    rttm_entries = read_rttm(rttm_file)
    vad_segments = read_vad_segments(vad_seg_file)
    combined_results = []
    
    for rttm in rttm_entries:
        rttm_filename = rttm['filename']
        rttm_start = rttm['start_time']
        rttm_end = rttm_start + rttm['duration']
        
        # Check overlap with VAD segments for the same file
        for vad in vad_segments:
            vad_filename = vad['filename']
            vad_start = vad['start_time']
            vad_end = vad['end_time']
            
            # Check if the filenames match and there is an overlap in time
            if vad_filename == rttm_filename and vad_end > rttm_start and vad_start < rttm_end:
                # Calculate the overlap time between VAD and RTTM
                overlap_start = max(vad_start, rttm_start)
                overlap_end = min(vad_end, rttm_end)
                duration = overlap_end - overlap_start

                if duration > 0:
                    # Append the aligned result
                    combined_results.append({
                        'filename': rttm_filename,
                        'channel': rttm['channel'],
                        'start_time': overlap_start,
                        'duration': duration,
                        'speaker_id': rttm['speaker_id']
                    })
    # Save the combined results to the specified RTTM file
    save_to_rttm(combined_results, output_rttm_file)
    print(f"Combined results saved to {output_rttm_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocessing RTTM file with VAD")
    parser.add_argument('--input_path', type=str, required=True, help='Input path: directory containing audio files')
    parser.add_argument('--input_rttm', type=str, required=True, help='Input RTTM file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--pyannote_segementation_token', type=str, required=True, help='Pyannote token')
    parser.add_argument('--output_rttm_file', type=str, required=True, help='Output RTTM file path')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate the wav.scp file
    wav_scp_file = f'{args.output_dir}/wav.scp'
    seg_vad_file = f'{args.output_dir}/seg'
    print(datetime.now().time(), ": Read all files to wav.scp")
    write_wav_scp(args.input_path, wav_scp_file)

    # Process VAD
    print(datetime.now().time(), ": Perform VAD")
    pyannote_vad(args.pyannote_segementation_token, wav_scp_file, seg_vad_file)
    print(datetime.now().time(), ": Refine RTTM with VAD")
    combine_vad_rttm(seg_vad_file, args.input_rttm, args.output_rttm_file)

    print(datetime.now().time(), f": Done postprocessing. Output is saved at {args.output_rttm_file}")
