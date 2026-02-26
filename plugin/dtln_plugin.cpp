#include <ladspa.h>
#include <fftw3.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdlib>

#define PORT_INPUT 0
#define PORT_OUTPUT 1

#define BLOCK_LEN 512
#define BLOCK_SHIFT 128

class DtlnProcessor {
public:
    LADSPA_Data* port_input;
    LADSPA_Data* port_output;
    float sample_rate;

    std::unique_ptr<tflite::FlatBufferModel> model_1;
    std::unique_ptr<tflite::FlatBufferModel> model_2;
    std::unique_ptr<tflite::Interpreter> interp_1;
    std::unique_ptr<tflite::Interpreter> interp_2;

    float in_buffer[BLOCK_LEN] = {0};
    float out_buffer[BLOCK_LEN] = {0};

    float* fft_in;
    fftwf_complex* fft_out;
    fftwf_plan plan_fft;
    
    fftwf_complex* ifft_in;
    float* ifft_out;
    fftwf_plan plan_ifft;

    float ring_in[4096] = {0};
    float ring_out[4096] = {0};
    int ring_in_head = 0, ring_in_tail = 0;
    int ring_out_head = 0, ring_out_tail = 0;

    DtlnProcessor(unsigned long s_rate) : sample_rate(s_rate) {
        fft_in = (float*)fftwf_malloc(sizeof(float) * BLOCK_LEN);
        fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (BLOCK_LEN / 2 + 1));
        plan_fft = fftwf_plan_dft_r2c_1d(BLOCK_LEN, fft_in, fft_out, FFTW_ESTIMATE);

        ifft_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (BLOCK_LEN / 2 + 1));
        ifft_out = (float*)fftwf_malloc(sizeof(float) * BLOCK_LEN);
        plan_ifft = fftwf_plan_dft_c2r_1d(BLOCK_LEN, ifft_in, ifft_out, FFTW_ESTIMATE);

        const char* m1_path = getenv("DTLN_MODEL_1") ? getenv("DTLN_MODEL_1") : "/usr/share/dtln/model_1_dynamic.tflite";
        const char* m2_path = getenv("DTLN_MODEL_2") ? getenv("DTLN_MODEL_2") : "/usr/share/dtln/model_2_dynamic.tflite";

        model_1 = tflite::FlatBufferModel::BuildFromFile(m1_path);
        model_2 = tflite::FlatBufferModel::BuildFromFile(m2_path);

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model_1, resolver)(&interp_1);
        tflite::InterpreterBuilder(*model_2, resolver)(&interp_2);

        // Required to utilize standard float32 I/O on Dynamic Range models
        interp_1->AllocateTensors();
        interp_2->AllocateTensors();
    }

    ~DtlnProcessor() {
        fftwf_destroy_plan(plan_fft);
        fftwf_destroy_plan(plan_ifft);
        fftwf_free(fft_in); fftwf_free(fft_out);
        fftwf_free(ifft_in); fftwf_free(ifft_out);
    }

    void process_128_samples(float* chunk_in, float* chunk_out) {
        memmove(in_buffer, in_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
        memcpy(in_buffer + (BLOCK_LEN - BLOCK_SHIFT), chunk_in, BLOCK_SHIFT * sizeof(float));

        memcpy(fft_in, in_buffer, BLOCK_LEN * sizeof(float));
        fftwf_execute(plan_fft);

        float mag[257];
        for (int i = 0; i < 257; ++i) {
            mag[i] = std::sqrt(fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1]);
        }

        // Setup dynamic indices
        int m1_mag_idx = -1, m1_state_idx = -1;
        for (int i = 0; i < interp_1->inputs().size(); ++i) {
            if (interp_1->tensor(interp_1->inputs()[i])->dims->data[interp_1->tensor(interp_1->inputs()[i])->dims->size - 1] == 257) {
                m1_mag_idx = i;
            } else {
                m1_state_idx = i;
            }
        }

        memcpy(interp_1->typed_input_tensor<float>(m1_mag_idx), mag, 257 * sizeof(float));
        interp_1->Invoke();

        int m1_out_mag_idx = -1, m1_out_state_idx = -1;
        for (int i = 0; i < interp_1->outputs().size(); ++i) {
            if (interp_1->tensor(interp_1->outputs()[i])->dims->data[interp_1->tensor(interp_1->outputs()[i])->dims->size - 1] == 257) {
                m1_out_mag_idx = i;
            } else {
                m1_out_state_idx = i;
            }
        }

        float* out_mag = interp_1->typed_output_tensor<float>(m1_out_mag_idx);
        float* out_states_1 = interp_1->typed_output_tensor<float>(m1_out_state_idx);

        memcpy(interp_1->typed_input_tensor<float>(m1_state_idx), out_states_1, interp_1->tensor(interp_1->inputs()[m1_state_idx])->bytes);

        for (int i = 0; i < 257; ++i) {
            float phase_angle = std::atan2(fft_out[i][1], fft_out[i][0]);
            ifft_in[i][0] = out_mag[i] * std::cos(phase_angle);
            ifft_in[i][1] = out_mag[i] * std::sin(phase_angle);
        }
        fftwf_execute(plan_ifft);

        for (int i = 0; i < BLOCK_LEN; ++i) {
            ifft_out[i] /= BLOCK_LEN;
        }

        int m2_feat_idx = -1, m2_state_idx = -1;
        for (int i = 0; i < interp_2->inputs().size(); ++i) {
            if (interp_2->tensor(interp_2->inputs()[i])->dims->data[interp_2->tensor(interp_2->inputs()[i])->dims->size - 1] == 512) {
                m2_feat_idx = i;
            } else {
                m2_state_idx = i;
            }
        }

        memcpy(interp_2->typed_input_tensor<float>(m2_feat_idx), ifft_out, BLOCK_LEN * sizeof(float));
        interp_2->Invoke();

        int m2_out_feat_idx = -1, m2_out_state_idx = -1;
        for (int i = 0; i < interp_2->outputs().size(); ++i) {
            if (interp_2->tensor(interp_2->outputs()[i])->dims->data[interp_2->tensor(interp_2->outputs()[i])->dims->size - 1] == 512) {
                m2_out_feat_idx = i;
            } else {
                m2_out_state_idx = i;
            }
        }

        float* out_time = interp_2->typed_output_tensor<float>(m2_out_feat_idx);
        float* out_states_2 = interp_2->typed_output_tensor<float>(m2_out_state_idx);
        
        memcpy(interp_2->typed_input_tensor<float>(m2_state_idx), out_states_2, interp_2->tensor(interp_2->inputs()[m2_state_idx])->bytes);

        for (int i = 0; i < BLOCK_LEN; ++i) {
            out_buffer[i] += out_time[i];
        }

        memcpy(chunk_out, out_buffer, BLOCK_SHIFT * sizeof(float));

        memmove(out_buffer, out_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
        memset(out_buffer + (BLOCK_LEN - BLOCK_SHIFT), 0, BLOCK_SHIFT * sizeof(float));
    }
};

LADSPA_Handle instantiate_dtln(const LADSPA_Descriptor* Descriptor, unsigned long SampleRate) {
    if (SampleRate != 16000) {
        std::cerr << "DTLN Warning: Host sample rate is " << SampleRate << ", but model expects 16000Hz!" << std::endl;
    }
    return new DtlnProcessor(SampleRate);
}

void connect_port_dtln(LADSPA_Handle Instance, unsigned long Port, LADSPA_Data* DataLocation) {
    DtlnProcessor* plugin = (DtlnProcessor*)Instance;
    if (Port == PORT_INPUT) plugin->port_input = DataLocation;
    else if (Port == PORT_OUTPUT) plugin->port_output = DataLocation;
}

void activate_dtln(LADSPA_Handle Instance) {
    DtlnProcessor* plugin = (DtlnProcessor*)Instance;
    memset(plugin->in_buffer, 0, sizeof(plugin->in_buffer));
    memset(plugin->out_buffer, 0, sizeof(plugin->out_buffer));
}

void run_dtln(LADSPA_Handle Instance, unsigned long SampleCount) {
    DtlnProcessor* plugin = (DtlnProcessor*)Instance;

    for (unsigned long i = 0; i < SampleCount; ++i) {
        plugin->ring_in[plugin->ring_in_head] = plugin->port_input[i];
        plugin->ring_in_head = (plugin->ring_in_head + 1) % 4096;
    }

    int available_in = (plugin->ring_in_head - plugin->ring_in_tail + 4096) % 4096;
    
    while (available_in >= BLOCK_SHIFT) {
        float chunk_in[BLOCK_SHIFT];
        float chunk_out[BLOCK_SHIFT];

        for (int i = 0; i < BLOCK_SHIFT; ++i) {
            chunk_in[i] = plugin->ring_in[plugin->ring_in_tail];
            plugin->ring_in_tail = (plugin->ring_in_tail + 1) % 4096;
        }

        plugin->process_128_samples(chunk_in, chunk_out);

        for (int i = 0; i < BLOCK_SHIFT; ++i) {
            plugin->ring_out[plugin->ring_out_head] = chunk_out[i];
            plugin->ring_out_head = (plugin->ring_out_head + 1) % 4096;
        }
        available_in -= BLOCK_SHIFT;
    }

    int available_out = (plugin->ring_out_head - plugin->ring_out_tail + 4096) % 4096;
    for (unsigned long i = 0; i < SampleCount; ++i) {
        if (available_out > 0) {
            plugin->port_output[i] = plugin->ring_out[plugin->ring_out_tail];
            plugin->ring_out_tail = (plugin->ring_out_tail + 1) % 4096;
            available_out--;
        } else {
            plugin->port_output[i] = 0.0f; 
        }
    }
}

void cleanup_dtln(LADSPA_Handle Instance) {
    delete (DtlnProcessor*)Instance;
}

LADSPA_Descriptor* g_descriptor = nullptr;

extern "C" const LADSPA_Descriptor* ladspa_descriptor(unsigned long Index) {
    if (Index != 0) return nullptr;

    if (!g_descriptor) {
        g_descriptor = new LADSPA_Descriptor;
        g_descriptor->UniqueID = 9999;
        g_descriptor->Label = "dtln_noise_suppression";
        g_descriptor->Properties = LADSPA_PROPERTY_HARD_RT_CAPABLE;
        g_descriptor->Name = "PiDTLN2 TFLite Neural Denoiser";
        g_descriptor->Maker = "Stuart Naylor";
        g_descriptor->Copyright = "MIT";
        g_descriptor->PortCount = 2;

        LADSPA_PortDescriptor* piPortDescriptors = new LADSPA_PortDescriptor[2];
        piPortDescriptors[PORT_INPUT] = LADSPA_PORT_AUDIO | LADSPA_PORT_INPUT;
        piPortDescriptors[PORT_OUTPUT] = LADSPA_PORT_AUDIO | LADSPA_PORT_OUTPUT;
        g_descriptor->PortDescriptors = piPortDescriptors;

        // FIX: Replaced fake type with standard const char** array
        const char** piPortNames = new const char*[2];
        piPortNames[PORT_INPUT] = "Audio In";
        piPortNames[PORT_OUTPUT] = "Audio Out";
        g_descriptor->PortNames = piPortNames;

        g_descriptor->instantiate = instantiate_dtln;
        g_descriptor->connect_port = connect_port_dtln;
        g_descriptor->activate = activate_dtln;
        g_descriptor->run = run_dtln;
        g_descriptor->run_adding = nullptr;
        g_descriptor->set_run_adding_gain = nullptr;
        g_descriptor->deactivate = nullptr;
        g_descriptor->cleanup = cleanup_dtln;
    }
    return g_descriptor;
}
