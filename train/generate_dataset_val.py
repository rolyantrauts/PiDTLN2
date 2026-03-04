import os
import json
import random
import tarfile
import zipfile
import io
import math
import shutil
import hashlib
import sys
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm
import tensorflow as tf
import multiprocessing
import pyarrow.parquet as pq
import gc

# --- CONFIGURATION ---
INDEX_FILE = "dataset_index.json"
OUTPUT_DIR = "tfrecords_dataset"
CACHE_DIR = "temp_cache_extracted"

TOTAL_HOURS = 600 
CLIP_DURATION = 4.0
SAMPLE_RATE = 16000
TARGET_PEAK = 0.8

# We keep shards relatively small (~2 hours per shard) for efficient streaming
NUM_SHARDS_TOTAL = 300 

# SAFETY: Leave 2 cores free.
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# SNR Settings
SNR_MIN = -5.0
SNR_MAX = 25.0
SNR_STEP = 2.5
SNR_OPTIONS = np.arange(SNR_MIN, SNR_MAX + SNR_STEP, SNR_STEP)

# TFRecord Settings
SAMPLES_PER_CLIP = int(CLIP_DURATION * SAMPLE_RATE)
CLIPS_PER_SHARD = int((TOTAL_HOURS * 3600 / CLIP_DURATION) / NUM_SHARDS_TOTAL)

# --- UTILS ---
def print_flush(msg):
    print(msg, flush=True)

def get_files_in_dir(path):
    """Recursively finds all audio files."""
    files = []
    if not os.path.exists(path): return []
    for r, d, f in os.walk(path):
        for file in f:
            if file.lower().endswith(('.wav', '.flac', '.mp3', '.opus')):
                files.append(os.path.join(r, file))
    return files

# ==========================================
# PHASE 1: ROBUST EXTRACTION
# ==========================================

def extract_archive(archive_path, target_dir, archive_type):
    """
    Extracts archive to target_dir. Returns count of files extracted.
    """
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    
    try:
        short_name = os.path.basename(archive_path)
        if len(short_name) > 20: short_name = short_name[:17] + "..."
        
        if archive_type == 'tar':
            with tarfile.open(archive_path, 'r') as t:
                members = []
                for m in t.getmembers():
                    if m.isfile() and m.name.lower().endswith(('.wav', '.flac', '.mp3')):
                        members.append(m)
                
                if not members: return 0
                
                for member in tqdm(members, desc=f"   Ext: {short_name}", unit="file", leave=False):
                    t.extract(member, path=target_dir)
                    count += 1

        elif archive_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as z:
                members = [n for n in z.namelist() if n.lower().endswith(('.wav', '.flac', '.mp3'))]
                if not members: return 0

                for member in tqdm(members, desc=f"   Ext: {short_name}", unit="file", leave=False):
                    z.extract(member, path=target_dir)
                    count += 1

        elif archive_type == 'parquet':
            table = pq.read_table(archive_path)
            col_names = table.column_names
            audio_col_name = next((c for c in col_names if 'audio' in c), None)
            
            if not audio_col_name:
                print_flush(f"   [!] No audio column found in {short_name}. Cols: {col_names}")
                return 0

            col = table[audio_col_name]
            
            for i in tqdm(range(len(table)), desc=f"   Ext: {short_name}", unit="row", leave=False):
                item = col[i].as_py()
                b_data = item.get('bytes') if isinstance(item, dict) else item
                
                if b_data:
                    with open(os.path.join(target_dir, f"row_{i}.wav"), "wb") as f:
                        f.write(b_data)
                    count += 1

    except Exception as e:
        print_flush(f"   [!] CRITICAL ERROR extracting {archive_path}: {e}")
        return 0
        
    return count

def prepare_sources(index_data):
    final_file_map = {}
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print_flush(f"--- PHASE 1: AUDITING & EXTRACTING SOURCES ---")

    grouped_index = {}
    for path, info in index_data.items():
        g = info['group']
        if g == 'noise': continue
        if g not in grouped_index: grouped_index[g] = []
        grouped_index[g].append(path)

    for group in sorted(grouped_index.keys()):
        print_flush(f"\n>>> Checking Group: {group}")
        paths = grouped_index[group]
        
        if group not in final_file_map: final_file_map[group] = []
        
        archives_to_process = []
        loose_files = []
        
        for path in paths:
            if not os.path.exists(path): continue

            if os.path.isdir(path):
                loose_files.extend(get_files_in_dir(path))
                for r, d, f in os.walk(path):
                    for file in f:
                        if file.lower().endswith(('.tar', '.tar.gz', '.zip', '.parquet')):
                            archives_to_process.append(os.path.join(r, file))
            else:
                if path.lower().endswith(('.tar', '.tar.gz', '.zip', '.parquet')):
                    archives_to_process.append(path)
                elif path.lower().endswith(('.wav', '.flac', '.mp3', '.opus')):
                    loose_files.append(path)

        if loose_files:
            final_file_map[group].extend(loose_files)
            print_flush(f"   - Index references {len(loose_files):,} existing loose files.")

        if archives_to_process:
            print_flush(f"   - Processing {len(archives_to_process)} archives...")
            
            for archive_path in archives_to_process:
                safe_name = os.path.basename(archive_path).replace('.', '_')
                path_hash = hashlib.md5(archive_path.encode()).hexdigest()[:6]
                target_dir = os.path.join(CACHE_DIR, group, f"{safe_name}_{path_hash}")
                marker_file = os.path.join(target_dir, "_completed.marker")
                
                needs_extraction = True
                if os.path.exists(marker_file):
                    existing = get_files_in_dir(target_dir)
                    if len(existing) > 0:
                        final_file_map[group].extend(existing)
                        needs_extraction = False
                    else:
                        print_flush(f"   [!] Found stale marker for {safe_name} but 0 files. Re-extracting.")
                        shutil.rmtree(target_dir)

                if needs_extraction:
                    atype = 'unknown'
                    if archive_path.endswith(('.tar', '.tar.gz')): atype = 'tar'
                    elif archive_path.endswith('.zip'): atype = 'zip'
                    elif archive_path.endswith('.parquet'): atype = 'parquet'
                    
                    if atype != 'unknown':
                        count = extract_archive(archive_path, target_dir, atype)
                        if count > 0:
                            with open(marker_file, 'w') as f: f.write("done")
                            final_file_map[group].extend(get_files_in_dir(target_dir))
                        else:
                            print_flush(f"   [!] Warning: 0 files extracted from {os.path.basename(archive_path)}")

        total_group = len(final_file_map[group])
        print_flush(f"   = TOTAL READY: {total_group:,} clips")

    total_all = sum(len(v) for v in final_file_map.values())
    print_flush(f"\n--- DATASET READY. Total Clips: {total_all:,} ---")
    return final_file_map

# ==========================================
# PHASE 2: GENERATION (WORKERS)
# ==========================================

global_noise_files = None

def worker_init(noise_files_list):
    global global_noise_files
    global_noise_files = noise_files_list

def process_audio(data, sr):
    if len(data.shape) > 1: data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE: data = resample_poly(data, SAMPLE_RATE, sr)
    
    if len(data) < SAMPLES_PER_CLIP:
        repeats = math.ceil(SAMPLES_PER_CLIP / len(data))
        data = np.tile(data, repeats)
    if len(data) > SAMPLES_PER_CLIP:
        start = random.randint(0, len(data) - SAMPLES_PER_CLIP)
        data = data[start : start + SAMPLES_PER_CLIP]
    else:
        data = data[:SAMPLES_PER_CLIP]
    return data.astype(np.float32)

def mix_audio(voice, noise, snr_db):
    voice_rms = np.sqrt(np.mean(voice**2))
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-9
    target_noise_rms = voice_rms / (10 ** (snr_db / 20))
    gain = target_noise_rms / noise_rms
    return voice + (noise * gain)

def simulate_agc_and_normalize(audio, pre_gain=1.5, forced_factor=None):
    audio_clipped = np.tanh(audio * pre_gain)
    
    if forced_factor is not None:
        return audio_clipped * forced_factor, forced_factor
        
    curr_peak = np.max(np.abs(audio_clipped))
    factor = TARGET_PEAK / curr_peak if curr_peak > 0 else 1.0
    return audio_clipped * factor, factor

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def worker_task(task_args):
    voice_path, snr = task_args
    try:
        v_data, v_sr = sf.read(voice_path)
        
        noise_path = random.choice(global_noise_files)
        n_data, n_sr = sf.read(noise_path)

        v_final = process_audio(v_data, v_sr)
        n_final = process_audio(n_data, n_sr)
        
        mixed = mix_audio(v_final, n_final, snr)
        
        final_input, agc_factor = simulate_agc_and_normalize(mixed)
        final_label, _ = simulate_agc_and_normalize(v_final, forced_factor=agc_factor)
        
        feature = {
            'audio': float_feature(final_input.flatten()),
            'label': float_feature(final_label.flatten())
        }
        
        # Explicitly delete large arrays to help the garbage collector
        del v_data, n_data, v_final, n_final, mixed
        
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    except Exception:
        return None

def process_task_list(task_list, output_dir, pool, desc_label):
    """
    Helper function to stream a list of tasks through the pool and write to shards.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    writer = tf.io.TFRecordWriter(os.path.join(output_dir, f"shard_{shard_idx:04d}.tfrecord"))
    clips_in_shard = 0
    
    iterator = pool.imap_unordered(worker_task, task_list, chunksize=10)
    
    for result in tqdm(iterator, total=len(task_list), desc=desc_label, unit="clips"):
        if result:
            writer.write(result)
            clips_in_shard += 1
            
            if clips_in_shard >= CLIPS_PER_SHARD:
                writer.close()
                shard_idx += 1
                clips_in_shard = 0
                writer = tf.io.TFRecordWriter(os.path.join(output_dir, f"shard_{shard_idx:04d}.tfrecord"))
                gc.collect() 
    
    writer.close()

# ==========================================
# MAIN DRIVER
# ==========================================

def main():
    multiprocessing.set_start_method('spawn', force=True)
    
    if not os.path.exists(INDEX_FILE):
        print_flush(f"Index file {INDEX_FILE} not found.")
        return

    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)

    print_flush("Scanning noise files...")
    noise_files = []
    noise_paths = [p for p, i in index_data.items() if i['group'] == 'noise']
    for p in noise_paths:
        if os.path.isdir(p):
            for r, d, f in os.walk(p):
                for file in f:
                    if file.lower().endswith('.wav'):
                        noise_files.append(os.path.join(r, file))
    print_flush(f"Noise files found: {len(noise_files)}")
    
    if not noise_files:
        print_flush("ERROR: No noise files found. Cannot proceed.")
        return

    file_map = prepare_sources(index_data)

    total_clips = int((TOTAL_HOURS * 3600) / CLIP_DURATION)
    print_flush(f"\nGoal: {TOTAL_HOURS} hours = {total_clips:,} clips.")
    
    group_counts = {g: len(files) for g, files in file_map.items()}
    total_avail = sum(group_counts.values())
    
    if total_avail == 0:
        print_flush("Error: No voice files found. Check your archives.")
        return

    task_list = []
    print_flush("\nTask Distribution:")
    for g, count in group_counts.items():
        if count == 0:
            continue
            
        ratio = count / total_avail
        needed = int(total_clips * ratio)
        print_flush(f"  {g:<15}: {needed:,} clips")
        
        available = file_map[g]
        for _ in range(needed):
            choice = random.choice(available)
            snr = random.choice(SNR_OPTIONS)
            task_list.append((choice, snr))
            
    # Perfect, randomized master list
    random.shuffle(task_list)
    print_flush(f"\nGenerated {len(task_list)} total tasks.")
    
    # 95:5 Split
    split_index = int(len(task_list) * 0.95)
    train_tasks = task_list[:split_index]
    val_tasks = task_list[split_index:]
    
    print_flush(f"Splitting Data: {len(train_tasks)} Train (95%) | {len(val_tasks)} Val (5%)")
    print_flush(f"Starting FAST POOL with {MAX_WORKERS} workers.")

    pool = multiprocessing.Pool(
        processes=MAX_WORKERS, 
        initializer=worker_init, 
        initargs=(noise_files,),
        maxtasksperchild=50 
    )
    
    try:
        # Process Training Data
        process_task_list(
            train_tasks, 
            os.path.join(OUTPUT_DIR, "train"), 
            pool, 
            desc_label="Generating TRAIN Data"
        )
        
        # Process Validation Data
        process_task_list(
            val_tasks, 
            os.path.join(OUTPUT_DIR, "val"), 
            pool, 
            desc_label="Generating VAL Data  "
        )
        
        pool.close()
        pool.join()
        print_flush(f"\nDone! Dataset saved to {OUTPUT_DIR}")

    except KeyboardInterrupt:
        pool.terminate()

if __name__ == "__main__":
    main()
