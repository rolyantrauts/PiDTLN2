```
sudo apt-get update
sudo apt-get install -y build-essential cmake git libfftw3-dev ladspa-sdk

# 1. Clone only the necessary version of TensorFlow (v2.15.1) 
# 2.15.1 was to match the DTLN TF with a slight bump before TF & Keras went seperate ways

cd ~
git clone --depth 1 -b v2.15.1 https://github.com/tensorflow/tensorflow.git

# 2. Create a build directory specifically for TFLite
mkdir -p tensorflow/tensorflow/lite/build
cd tensorflow/tensorflow/lite/build

# 3. Configure and build the static library
cmake ..
cmake --build . 

cd ~
mkdir -p PiDTLN2/plugin/build
cd build
cmake ..
make

# Install the plugin
sudo cp dtln_ladspa.so /usr/lib/ladspa/
sudo chmod 644 /usr/lib/ladspa/dtln_ladspa.so

# Install the models
sudo mkdir -p /usr/share/dtln
sudo cp ../../models/model_1_dynamic.tflite /usr/share/dtln/
sudo cp ../../models/model_2_dynamic.tflite /usr/share/dtln/
sudo chmod 644 /usr/share/dtln/*.tflite

# test ladspa plugin
ldd /usr/lib/ladspa/dtln_ladspa.so
analyseplugin /usr/lib/ladspa/dtln_ladspa.so

nano ~/.asoundrc

pcm.dtln_mic {
    type ladspa
    slave.pcm "hw:0,0"
    path "/usr/lib/ladspa"
    plugins [{
        label dtln_noise_suppression
        id 39401
    }]
}

arecord -Ddtln_mic -r16000 -fS16_LE -c1 test.wav
```
