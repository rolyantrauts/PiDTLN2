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

# Configure and optimize specifically for the Cortex-A53
# cmake -DCMAKE_C_FLAGS="-O3 -mcpu=cortex-a53 -mtune=cortex-a53" \
#       -DCMAKE_CXX_FLAGS="-O3 -mcpu=cortex-a53 -mtune=cortex-a53" \
#       -DTFLITE_ENABLE_XNNPACK=ON \
#       -DTFLITE_ENABLE_RUY=ON \
#       ..

cd ~
mkdir -p PiDTLN2/plugin/build
cd PiDTLN2/plugin/build
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
    slave.pcm "plughw:0,0"
    path "/usr/lib/ladspa"
    plugins [{
        label dtln_noise_suppression
        id 9999 # Change in dtln_plugin.cpp
    }]
}

arecord -Dplug:dtln_mic -r16000 -fS16_LE -c1 test.wav

# GStreamer pipeline
gst-launch-1.0 alsasrc ! \
    audio/x-raw,rate=16000,channels=1,format=S16LE ! \
    audioconvert ! \
    ladspa-speech-agc-neon-wakeword-agc-neon \
    Target-Level--Amplitude-=0.5 \
    Attack-Time--ms-=0.1 \
    Release-Time--ms-=1550.0 \
    Hold-Time--ms-=400.0 ! \
    audioconvert ! \
    alsasink```
