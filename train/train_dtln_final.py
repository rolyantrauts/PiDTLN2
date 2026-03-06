import tensorflow as tf
import os
import glob
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- CONFIGURATION ---
TFRECORD_TRAIN_DIR = "tfrecords_dataset/train"
TFRECORD_VAL_DIR = "tfrecords_dataset/val"

# Increased for 16-core Xeon with 64GB RAM
BATCH_SIZE = 64  
EPOCHS = 200

# Zero-indexed. Setting to 44 starts training at Epoch 45/200
INITIAL_EPOCH = 44  
LEARNING_RATE = 1e-3
MODEL_SAVE_DIR = "saved_models"

# 10 seconds at 16kHz to match the new dataset generator
CLIP_LENGTH = 160000  

# DTLN Hyperparameters (from official repo)
BLOCK_LEN = 512
BLOCK_SHIFT = 128
NUM_UNITS = 128
NUM_LAYERS = 2
ENCODER_SIZE = 256
DROPOUT = 0.25

# ==========================================
# TFRECORD DATA PIPELINE
# ==========================================

def parse_tfrecord(example_proto):
    feature_description = {
        'audio': tf.io.VarLenFeature(tf.float32), 
        'label': tf.io.VarLenFeature(tf.float32)  
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    audio = tf.sparse.to_dense(parsed['audio'])
    label = tf.sparse.to_dense(parsed['label'])
    
    audio = tf.reshape(audio, [CLIP_LENGTH])
    label = tf.reshape(label, [CLIP_LENGTH])
    
    return audio, label

def get_dataset(tfrecord_dir, batch_size, is_training=True):
    # list_files randomizes the order of the shards.
    files_pattern = os.path.join(tfrecord_dir, "*.tfrecord")
    dataset = tf.data.Dataset.list_files(files_pattern, shuffle=is_training)
    
    # Interleave reads from multiple files simultaneously for true global randomness
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training
    )
    
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Deep shuffle using available 64GB RAM for superior batch-level randomization
    if is_training:
        dataset = dataset.shuffle(buffer_size=8000, reshuffle_each_iteration=True)
        
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# ==========================================
# DTLN CUSTOM LAYERS & LOSS
# ==========================================

def snr_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    
    epsilon = 1e-7
    
    # Calculate energy of true signal and noise
    true_energy = tf.reduce_mean(tf.math.square(y_true), axis=-1, keepdims=True)
    noise_energy = tf.reduce_mean(tf.math.square(y_true - y_pred), axis=-1, keepdims=True)
    
    # Ensure inputs to log are strictly positive and bounded
    true_energy = tf.maximum(true_energy, epsilon)
    noise_energy = tf.maximum(noise_energy, epsilon)
    
    # log(A/B) = log(A) - log(B) is more numerically stable than log(A / B)
    # Using base 10 log: log10(x) = ln(x) / ln(10)
    log_10 = tf.math.log(10.0)
    
    log_true = tf.math.log(true_energy) / log_10
    log_noise = tf.math.log(noise_energy) / log_10
    
    # SNR = 10 * log10(True / Noise)
    # We negate it because we want to MINIMIZE the loss (maximize SNR)
    loss = -10.0 * (log_true - log_noise)
    
    return tf.reduce_mean(loss)

class InstantLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7

    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma = self.add_weight(shape=shape, initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=shape, initializer='zeros', trainable=True, name='beta')

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma + self.beta
        return outputs

def stft_layer(x):
    frames = tf.signal.frame(x, BLOCK_LEN, BLOCK_SHIFT)
    stft_dat = tf.signal.rfft(frames)
    mag = tf.abs(stft_dat)
    phase = tf.math.angle(stft_dat)
    return [mag, phase]

def ifft_layer(x):
    mag, phase = x[0], x[1]
    
    real_part = mag * tf.math.cos(phase)
    imag_part = mag * tf.math.sin(phase)
    
    s1_stft = tf.complex(real_part, imag_part)
    return tf.signal.irfft(s1_stft)

def overlap_add_layer(x):
    return tf.signal.overlap_and_add(x, BLOCK_SHIFT)

def separation_kernel(num_layer, mask_size, x):
    for idx in range(num_layer):
        x = tf.keras.layers.LSTM(NUM_UNITS, return_sequences=True)(x)
        if idx < (num_layer - 1):
            x = tf.keras.layers.Dropout(DROPOUT)(x)
            
    mask = tf.keras.layers.Dense(mask_size, activation='sigmoid')(x)
    return mask

# ==========================================
# BUILD THE ACTUAL DTLN MODEL
# ==========================================

def build_dtln_model():
    time_dat = tf.keras.Input(shape=(CLIP_LENGTH,), name="input_audio")
    
    mag, angle = tf.keras.layers.Lambda(stft_layer, name="stft")(time_dat)
    mag_norm = mag 
    
    mask_1 = separation_kernel(NUM_LAYERS, (BLOCK_LEN // 2 + 1), mag_norm)
    estimated_mag = tf.keras.layers.Multiply(name="multiply_core1")([mag, mask_1])
    
    estimated_frames_1 = tf.keras.layers.Lambda(ifft_layer, name="ifft")([estimated_mag, angle])
    
    encoded_frames = tf.keras.layers.Conv1D(ENCODER_SIZE, 1, strides=1, use_bias=False, name="conv1d_encode")(estimated_frames_1)
    encoded_frames_norm = InstantLayerNormalization(name="inst_layer_norm")(encoded_frames)
    
    mask_2 = separation_kernel(NUM_LAYERS, ENCODER_SIZE, encoded_frames_norm)
    estimated = tf.keras.layers.Multiply(name="multiply_core2")([encoded_frames, mask_2])
    
    decoded_frames = tf.keras.layers.Conv1D(BLOCK_LEN, 1, padding='causal', use_bias=False, name="conv1d_decode")(estimated)
    
    estimated_sig = tf.keras.layers.Lambda(overlap_add_layer, name="overlap_add")(decoded_frames)
    
    return tf.keras.Model(inputs=time_dat, outputs=estimated_sig)

# ==========================================
# MAIN ROUTINE
# ==========================================

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print("Initializing TFRecord Dataset Pipelines...")
    train_dataset = get_dataset(TFRECORD_TRAIN_DIR, BATCH_SIZE, is_training=True)
    val_dataset = get_dataset(TFRECORD_VAL_DIR, BATCH_SIZE, is_training=False)
    
    print("Building Official DTLN Architecture...")
    model = build_dtln_model()
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE, clipnorm=3.0)
    
    model.compile(optimizer=optimizer, loss=snr_loss)

    # Load the healthy BEST validation weights
    best_val_path = os.path.join(MODEL_SAVE_DIR, "dtln_best_val.keras")
    if os.path.exists(best_val_path):
        print(f"\n[INFO] Found healthy saved model at {best_val_path}. Loading weights to resume training...")
        model.load_weights(best_val_path)
    else:
        print("\n[INFO] No previous checkpoint found. Starting training from scratch...")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "dtln_best_val.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "dtln_last_epoch.keras"),
            save_best_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            cooldown=1,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]
    
    print(f"\nStarting DTLN Training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=INITIAL_EPOCH,  
        epochs=EPOCHS, 
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
