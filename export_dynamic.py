import os
import tensorflow as tf

# Suppress verbose TF logging during conversion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def get_all_layers(model):
    """Recursively extract all layers from a nested Keras model."""
    layers = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            layers.extend(get_all_layers(layer))
        else:
            layers.append(layer)
    return layers

def build_stateful_model_1(num_units=128, num_layers=2):
    """Stage 1: Magnitude Domain (Shape: 257) with Exposed States"""
    mag_in = tf.keras.Input(batch_shape=(1, 1, 257), name='mag_in')
    states_in = tf.keras.Input(batch_shape=(1, num_layers, num_units, 2), name='states_1_in')
    
    # Exact math match to original training (1e-7)
    x = tf.math.log(mag_in + 1e-7)
    x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-7, name='norm_1')(x)
    
    states_out_h = []
    states_out_c = []
    
    for idx in range(num_layers):
        in_state_h = states_in[:, idx, :, 0]
        in_state_c = states_in[:, idx, :, 1]
        
        # unroll=True forces static math for edge deployment
        x, out_h, out_c = tf.keras.layers.LSTM(
            num_units, 
            return_sequences=True, 
            return_state=True, 
            unroll=True, 
            name=f'lstm_{idx}'
        )(x, initial_state=[in_state_h, in_state_c])
        
        states_out_h.append(out_h)
        states_out_c.append(out_c)
        
    mask = tf.keras.layers.Dense(257, activation='sigmoid', name='dense_1')(x)
    estimated_mag = tf.keras.layers.Multiply(name='multiply_1')([mag_in, mask])
    
    states_out_h = tf.stack(states_out_h, axis=1)
    states_out_c = tf.stack(states_out_c, axis=1)
    states_out = tf.stack([states_out_h, states_out_c], axis=-1, name='states_1_out')
    
    return tf.keras.Model(inputs=[mag_in, states_in], outputs=[estimated_mag, states_out])

def build_stateful_model_2(num_units=128, num_layers=2, block_len=512):
    """Stage 2: Time/Feature Domain (Shape: 256) with Exposed States"""
    feat_in = tf.keras.Input(batch_shape=(1, 1, block_len), name='feat_in')
    states_in = tf.keras.Input(batch_shape=(1, num_layers, num_units, 2), name='states_2_in')
    
    x = tf.keras.layers.Conv1D(256, 1, strides=1, use_bias=False, name='conv1d_1')(feat_in)
    x_lstm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-7, name='norm_2')(x)
        
    states_out_h = []
    states_out_c = []
    
    for idx in range(num_layers):
        in_state_h = states_in[:, idx, :, 0]
        in_state_c = states_in[:, idx, :, 1]
        
        # unroll=True forces static math for edge deployment
        x_lstm, out_h, out_c = tf.keras.layers.LSTM(
            num_units, 
            return_sequences=True, 
            return_state=True, 
            unroll=True, 
            name=f'lstm_{idx + 2}'
        )(x_lstm, initial_state=[in_state_h, in_state_c])
        
        states_out_h.append(out_h)
        states_out_c.append(out_c)
        
    mask = tf.keras.layers.Dense(256, activation='sigmoid', name='dense_2')(x_lstm)
    estimated_feat = tf.keras.layers.Multiply(name='multiply_2')([x, mask])
    out = tf.keras.layers.Conv1D(block_len, 1, strides=1, use_bias=False, name='conv1d_2')(estimated_feat)
    
    states_out_h = tf.stack(states_out_h, axis=1)
    states_out_c = tf.stack(states_out_c, axis=1)
    states_out = tf.stack([states_out_h, states_out_c], axis=-1, name='states_2_out')
    
    return tf.keras.Model(inputs=[feat_in, states_in], outputs=[out, states_out])

def export_dynamic_range_models(saved_model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nLoading original SavedModel from {saved_model_path}...")
    original_model = tf.keras.models.load_model(saved_model_path, compile=False)
    layers = get_all_layers(original_model)
    
    lstms = [l for l in layers if 'lstm' in l.__class__.__name__.lower()]
    denses = [l for l in layers if 'dense' in l.__class__.__name__.lower()]
    convs = [l for l in layers if 'conv' in l.__class__.__name__.lower()]
    norms = [l for l in layers if 'normalization' in l.__class__.__name__.lower()]
    
    print("Building state-exposing static (unrolled) Models...")
    m1 = build_stateful_model_1()
    m2 = build_stateful_model_2()
    
    # Map extracted weights perfectly into the new unrolled, state-exposing structures
    m1.get_layer('norm_1').set_weights(norms[0].get_weights())
    m1.get_layer('lstm_0').set_weights(lstms[0].get_weights())
    m1.get_layer('lstm_1').set_weights(lstms[1].get_weights())
    m1.get_layer('dense_1').set_weights(denses[0].get_weights())
    
    m2.get_layer('conv1d_1').set_weights(convs[0].get_weights())
    m2.get_layer('norm_2').set_weights(norms[1].get_weights())
    m2.get_layer('lstm_2').set_weights(lstms[2].get_weights())
    m2.get_layer('lstm_3').set_weights(lstms[3].get_weights())
    m2.get_layer('dense_2').set_weights(denses[1].get_weights())
    m2.get_layer('conv1d_2').set_weights(convs[1].get_weights())
    
    def convert_and_save(keras_model, filename):
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        # Dynamic Range Quantization Configuration
        # This converts the heavy LSTM and Dense weights to INT8, 
        # but keeps activations and hidden states in Float32 to avoid precision collapse.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        print(f"Exported Dynamic Range TFLite model to: {filepath}")

    print("\nQuantizing Model 1 (Magnitude Network)...")
    convert_and_save(m1, "model_1_dynamic.tflite")
    
    print("\nQuantizing Model 2 (Feature Network)...")
    convert_and_save(m2, "model_2_dynamic.tflite")
    print("\nExtraction and Export Complete.")

if __name__ == "__main__":
    # Adjust paths to your environment
    SAVED_MODEL_DIR = "./pretrained_model/DTLN_norm_500h_saved_model"
    OUTPUT_DIR = "./export_int8"
    
    if os.path.exists(SAVED_MODEL_DIR):
        export_dynamic_range_models(SAVED_MODEL_DIR, OUTPUT_DIR)
    else:
        print(f"Error: Could not find Model directory at {SAVED_MODEL_DIR}.")