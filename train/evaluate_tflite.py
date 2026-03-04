import os
import time
import numpy as np
import soundfile as sf
import tensorflow as tf

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def calculate_si_sdr(reference, estimation):
    """Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)"""
    # Remove DC offset
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Trim to matching lengths
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # Calculate target and noise
    target = np.sum(reference * estimation) / (np.sum(reference**2) + 1e-8) * reference
    noise = estimation - target
    
    # Calculate SI-SDR
    sdr = 10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + 1e-8) + 1e-8)
    return sdr

def get_tensor_indices_by_shape(interpreter, target_last_dim):
    """Dynamically finds tensor indices based on the expected last dimension shape."""
    feat_idx, state_idx = -1, -1
    
    for detail in interpreter.get_input_details():
        if detail['shape'][-1] == target_last_dim:
            feat_idx = detail['index']
        else:
            state_idx = detail['index']
            
    out_feat_idx, out_state_idx = -1, -1
    for detail in interpreter.get_output_details():
        if detail['shape'][-1] == target_last_dim:
            out_feat_idx = detail['index']
        else:
            out_state_idx = detail['index']
            
    return feat_idx, state_idx, out_feat_idx, out_state_idx

def evaluate_batch(noisy_dir, clean_dir, out_dir, m1_path, m2_path):
    print(f"{'='*60}")
    print("Batch TFLite Inference & Phase-Aligned SI-SDR Evaluation")
    print(f"{'='*60}\n")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading TFLite Interpreters...")
    interp_1 = tf.lite.Interpreter(model_path=m1_path)
    interp_1.allocate_tensors()
    interp_2 = tf.lite.Interpreter(model_path=m2_path)
    interp_2.allocate_tensors()

    m1_in_mag, m1_in_states, m1_out_mag, m1_out_states = get_tensor_indices_by_shape(interp_1, 257)
    m2_in_feat, m2_in_states, m2_out_feat, m2_out_states = get_tensor_indices_by_shape(interp_2, 512)

    block_len = 512
    block_shift = 128
    
    # Calculate algorithmic delay based on OLA framing
    algorithmic_delay = block_len - block_shift
    
    total_baseline_sdr = 0.0
    total_processed_sdr = 0.0
    processed_files = 0

    print("\nStarting Batch Processing...")
    print(f"{'File':<18} | {'Noisy SDR':<10} | {'Clean SDR':<10} | {'Improvement':<10}")
    print("-" * 55)

    for i in range(20):
        filename = f"example_{i:03d}.wav"
        noisy_wav = os.path.join(noisy_dir, filename)
        clean_wav = os.path.join(clean_dir, filename)
        out_wav = os.path.join(out_dir, filename)
        
        if not os.path.exists(noisy_wav) or not os.path.exists(clean_wav):
            continue
            
        audio_in, fs = sf.read(noisy_wav)
        clean_audio, _ = sf.read(clean_wav)
        
        if len(audio_in.shape) > 1: audio_in = audio_in[:, 0]
        if len(clean_audio.shape) > 1: clean_audio = clean_audio[:, 0]

        in_buffer = np.zeros(block_len, dtype=np.float32)
        out_buffer = np.zeros(block_len, dtype=np.float32)
        out_audio = np.zeros_like(audio_in)
        
        states_1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        states_2 = np.zeros((1, 2, 128, 2), dtype=np.float32)

        num_blocks = (len(audio_in) - algorithmic_delay) // block_shift
        
        for b in range(num_blocks):
            idx = b * block_shift
            
            in_buffer = np.roll(in_buffer, -block_shift)
            in_buffer[-block_shift:] = audio_in[idx : idx + block_shift]

            fft_out = np.fft.rfft(in_buffer)
            mag = np.abs(fft_out).astype(np.float32).reshape(1, 1, 257)
            phase = np.angle(fft_out)

            interp_1.set_tensor(m1_in_mag, mag)
            interp_1.set_tensor(m1_in_states, states_1)
            interp_1.invoke()
            out_mag = interp_1.get_tensor(m1_out_mag)
            states_1 = interp_1.get_tensor(m1_out_states)

            estimated_complex = out_mag.flatten() * np.exp(1j * phase)
            ifft_out = np.fft.irfft(estimated_complex).astype(np.float32).reshape(1, 1, 512)

            interp_2.set_tensor(m2_in_feat, ifft_out)
            interp_2.set_tensor(m2_in_states, states_2)
            interp_2.invoke()
            out_time = interp_2.get_tensor(m2_out_feat)
            states_2 = interp_2.get_tensor(m2_out_states)

            out_buffer += out_time.flatten()
            out_audio[idx : idx + block_shift] = out_buffer[:block_shift]
            
            out_buffer = np.roll(out_buffer, -block_shift)
            out_buffer[-block_shift:] = 0.0

        # --- LATENCY COMPENSATION ---
        # Shift the audio back by the algorithmic delay to perfectly align with the clean reference
        aligned_out_audio = np.zeros_like(out_audio)
        aligned_out_audio[:-algorithmic_delay] = out_audio[algorithmic_delay:]

        sf.write(out_wav, aligned_out_audio, 16000)
        
        b_sdr = calculate_si_sdr(clean_audio, audio_in)
        p_sdr = calculate_si_sdr(clean_audio, aligned_out_audio)
        improvement = p_sdr - b_sdr
        
        total_baseline_sdr += b_sdr
        total_processed_sdr += p_sdr
        processed_files += 1
        
        print(f"{filename:<18} | {b_sdr:>7.2f} dB | {p_sdr:>7.2f} dB | {improvement:>7.2f} dB")

    if processed_files > 0:
        avg_baseline = total_baseline_sdr / processed_files
        avg_processed = total_processed_sdr / processed_files
        avg_improvement = avg_processed - avg_baseline
        
        print("-" * 55)
        print(f"{'AVERAGE':<18} | {avg_baseline:>7.2f} dB | {avg_processed:>7.2f} dB | {avg_improvement:>7.2f} dB")
        print(f"\nSuccessfully processed {processed_files} files.")
    else:
        print("\nNo files were processed. Check your directory paths.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    NOISY_DIR = "./noisy_input"
    CLEAN_DIR = "./clean_reference"
    OUT_DIR = "./processed_output"
    
    # Note: Running against the Float32 models or INT8 models will both work here
    MODEL_1 = "./export_int8/model_1_dynamic.tflite"
    MODEL_2 = "./export_int8/model_2_dynamic.tflite"
    
    evaluate_batch(NOISY_DIR, CLEAN_DIR, OUT_DIR, MODEL_1, MODEL_2)