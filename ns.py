#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process realtime audio with a trained DTLN model. 
This script supports ALSA audio devices. The model expects 16kHz single channel audio input/output.

PiDTLN2 Update:
- Supports Dynamic Range Quantized models (Float32 I/O, INT8 internal weights).
- Dynamically resolves tensor indices to support unrolled architectures.
- Eliminates manual mask multiplication (Multiply layer is baked into Model 1).

Example call:
    $python ns.py -i capture -o playback

Author: sanebow (sanebow@gmail.com)
PiDTLN2 Mod: Stuart Naylor
Version: 2026

This code is licensed under the terms of the MIT-license.
"""

import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import argparse
import collections
import time
import daemon
import threading

g_use_fftw = True

try:
    import pyfftw
except ImportError:
    print("[WARNING] pyfftw is not installed, use np.fft")
    g_use_fftw = False

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channel', type=int, default=None,
    help='use specific channel of input device')
parser.add_argument(
    '-n', '--no-denoise', action='store_true',
    help='turn off denoise, pass-through')
parser.add_argument(
    '-t', '--threads', type=int, default=1,
    help='number of threads for tflite interpreters')
parser.add_argument(
    '--latency', type=float, default=0.2,
    help='suggested input/output latency in seconds')
parser.add_argument(
    '-D', '--daemonize', action='store_true',
    help='run as a daemon')
parser.add_argument(
    '--measure', action='store_true',
    help='measure and report processing time')
parser.add_argument(
    '--no-fftw', action='store_true',
    help='use np.fft instead of fftw')

args = parser.parse_args(remaining)

# Set some parameters
block_len_ms = 32 
block_shift_ms = 8
fs_target = 16000

# Create the interpreters (Targeting the new Dynamic Range TFLite models)
interpreter_1 = tflite.Interpreter(model_path='./models/model_1_dynamic.tflite', num_threads=args.threads)
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./models/model_2_dynamic.tflite', num_threads=args.threads)
interpreter_2.allocate_tensors()

# Get input and output tensors
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

# Dynamically resolve indices to handle unrolled/re-exported graphs
m1_mag_idx = next(d['index'] for d in input_details_1 if d['shape'][-1] == 257)
m1_in_state_idx = next(d['index'] for d in input_details_1 if len(d['shape']) == 4)
m1_out_mag_idx = next(d['index'] for d in output_details_1 if d['shape'][-1] == 257)
m1_out_state_idx = next(d['index'] for d in output_details_1 if len(d['shape']) == 4)

m2_feat_idx = next(d['index'] for d in input_details_2 if d['shape'][-1] == 512)
m2_in_state_idx = next(d['index'] for d in input_details_2 if len(d['shape']) == 4)
m2_out_feat_idx = next(d['index'] for d in output_details_2 if d['shape'][-1] == 512)
m2_out_state_idx = next(d['index'] for d in output_details_2 if len(d['shape']) == 4)

# Create states for the LSTMs based on dynamic shapes
states_1 = np.zeros(next(d['shape'] for d in input_details_1 if d['index'] == m1_in_state_idx)).astype('float32')
states_2 = np.zeros(next(d['shape'] for d in input_details_2 if d['index'] == m2_in_state_idx)).astype('float32')

# Calculate shift and length
block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
block_len = int(np.round(fs_target * (block_len_ms / 1000)))

# Create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')

if args.no_fftw:
    g_use_fftw = False
if g_use_fftw:
    fft_buf = pyfftw.empty_aligned(512, dtype='float32')
    rfft = pyfftw.builders.rfft(fft_buf)
    ifft_buf = pyfftw.empty_aligned(257, dtype='complex64')
    irfft = pyfftw.builders.irfft(ifft_buf)

t_ring = collections.deque(maxlen=100)

def callback(indata, outdata, frames, buf_time, status):
    global in_buffer, out_buffer, states_1, states_2, t_ring, g_use_fftw
    
    if args.measure:
        start_time = time.time()
    if status:
        print(status)
    if args.channel is not None:
        indata = indata[:, [args.channel]] 
        
    if args.no_denoise:
        outdata[:] = indata
        if args.measure:
            t_ring.append(time.time() - start_time)
        return
        
    # Write new samples to input buffer (Shift Register)
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata)
    
    # Calculate FFT of input block
    if g_use_fftw:
        fft_buf[:] = in_buffer
        in_block_fft = rfft()
    else:
        in_block_fft = np.fft.rfft(in_buffer)
        
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    
    # Reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
    
    # Set tensors to the first model
    interpreter_1.set_tensor(m1_in_state_idx, states_1)
    interpreter_1.set_tensor(m1_mag_idx, in_mag)
    
    # Run calculation for Stage 1
    interpreter_1.invoke()
    
    # Get the multiplied magnitude output and updated states
    out_mag = interpreter_1.get_tensor(m1_out_mag_idx) 
    states_1 = interpreter_1.get_tensor(m1_out_state_idx)   
    
    # Calculate the iFFT (PiDTLN2 models already apply the mask multiplication internally)
    estimated_complex = out_mag * np.exp(1j * in_phase)
    
    if g_use_fftw:
        ifft_buf[:] = np.squeeze(estimated_complex)
        estimated_block = irfft()
    else:
        estimated_block = np.fft.irfft(estimated_complex)
        
    # Reshape the time domain block for Stage 2
    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')
    
    # Set tensors to the second model
    interpreter_2.set_tensor(m2_in_state_idx, states_2)
    interpreter_2.set_tensor(m2_feat_idx, estimated_block)
    
    # Run calculation for Stage 2
    interpreter_2.invoke()
    
    # Get output block and updated states
    out_block = interpreter_2.get_tensor(m2_out_feat_idx) 
    states_2 = interpreter_2.get_tensor(m2_out_state_idx) 
    
    # Write to output Overlap-Add buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)
    
    # Output perfectly aligned 128 samples to soundcard
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)
    
    if args.measure:
        t_ring.append(time.time() - start_time)

def open_stream():
    with sd.Stream(device=(args.input_device, args.output_device),
                samplerate=fs_target, blocksize=block_shift,
                dtype=np.float32, latency=args.latency,
                channels=(1 if args.channel is None else None, 1), callback=callback):
        print('#' * 80)
        print('PiDTLN2 Real-Time Noise Suppression Active')
        print('Ctrl-C to exit')
        print('#' * 80)
        if args.measure:
            while True:
                time.sleep(1)
                print('Processing time: {:.2f} ms'.format( 1000 * np.average(t_ring) ), end='\r')
        else:
            threading.Event().wait()

try:
    if args.daemonize:
        with daemon.DaemonContext():
            open_stream()
    else:
        open_stream()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
