// SPDX-FileCopyrightText: Copyright (c) 2025-2026
// SPDX-License-Identifier: Apache-2.0
//
// C++ Port of TensorRT Edge-LLM Offline Audio Preprocessing
// Uses FFTW3 for extreme performance. Matches WhisperFeatureExtractor.

#include "bot_speech/audio_preprocessor.hpp"

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace bot_speech
{

AudioPreprocessor::AudioPreprocessor() {}

float AudioPreprocessor::hz_to_mel(float f) {
    float min_log_hz = 1000.0f;
    if (f < min_log_hz) return f * 3.0f / 200.0f;
    return 15.0f + 27.5664f * std::log10(f / 1000.0f);
}

float AudioPreprocessor::mel_to_hz(float mel) {
    if (mel < 15.0f) return mel * 200.0f / 3.0f;
    return 1000.0f * std::pow(10.0f, (mel - 15.0f) / 27.5664f);
}

uint16_t AudioPreprocessor::float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint16_t sign = (bits >> 16) & 0x8000;
    int16_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint16_t mantissa = (bits >> 13) & 0x3FF;

    if (exponent <= 0) return sign;
    if (exponent > 31) return sign | 0x7C00;
    return sign | (exponent << 10) | mantissa;
}

std::vector<std::vector<float>> AudioPreprocessor::create_mel_filterbank() {
    std::vector<float> fft_freqs(n_fft_ / 2 + 1);
    for (int i = 0; i <= n_fft_ / 2; i++) {
        fft_freqs[i] = static_cast<float>(i * sample_rate_) / n_fft_;
    }

    float mel_min = hz_to_mel(f_min_);
    float mel_max = hz_to_mel(f_max_);
    std::vector<float> mel_points(n_mels_ + 2);
    for (int i = 0; i < n_mels_ + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (n_mels_ + 1);
        mel_points[i] = mel_to_hz(mel);
    }

    std::vector<std::vector<float>> filterbank(n_mels_, std::vector<float>(n_fft_ / 2 + 1, 0.0f));

    for (int m = 0; m < n_mels_; m++) {
        for (int k = 0; k <= n_fft_ / 2; k++) {
            float f = fft_freqs[k];
            if (f >= mel_points[m] && f <= mel_points[m + 1]) {
                filterbank[m][k] = (f - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
            } else if (f > mel_points[m + 1] && f <= mel_points[m + 2]) {
                filterbank[m][k] = (mel_points[m + 2] - f) / (mel_points[m + 2] - mel_points[m + 1]);
            }
        }
        float enorm = 2.0f / (mel_points[m + 2] - mel_points[m]);
        for (int k = 0; k <= n_fft_ / 2; k++) {
            filterbank[m][k] *= enorm;
        }
    }
    return filterbank;
}

std::vector<float> AudioPreprocessor::load_wav(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open WAV file.");

    char riff[4]; file.read(riff, 4);
    file.seekg(22, std::ios::beg);
    uint16_t channels; file.read(reinterpret_cast<char*>(&channels), 2);
    uint32_t sr; file.read(reinterpret_cast<char*>(&sr), 4);

    if (sr != 16000) std::cerr << "Warning: Audio is not 16000Hz!" << std::endl;

    file.seekg(34, std::ios::beg);
    uint16_t bits; file.read(reinterpret_cast<char*>(&bits), 2);

    uint32_t data_size = 0;
    file.seekg(36, std::ios::beg);
    while (true) {
        char chunk_id[4]; file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&data_size), 4);
        if (std::strncmp(chunk_id, "data", 4) == 0) break;
        file.seekg(data_size, std::ios::cur);
    }

    int num_samples = data_size / (channels * (bits / 8));
    std::vector<float> audio(num_samples);

    if (bits == 16) {
        std::vector<int16_t> raw(num_samples * channels);
        file.read(reinterpret_cast<char*>(raw.data()), data_size);
        for (int i = 0; i < num_samples; i++) audio[i] = raw[i * channels] / 32768.0f;
    } else if (bits == 32) {
        std::vector<float> raw(num_samples * channels);
        file.read(reinterpret_cast<char*>(raw.data()), data_size);
        for (int i = 0; i < num_samples; i++) audio[i] = raw[i * channels];
    } else {
        throw std::runtime_error("Unsupported WAV bit depth. Use 16 or 32 bits.");
    }
    return audio;
}

std::vector<float> AudioPreprocessor::extract_whisper_mel(
    const std::vector<float>& audio, int& out_time_steps) {
    int time_steps = (audio.size() - n_fft_) / hop_length_ + 1;
    out_time_steps = time_steps;
    if (time_steps <= 0) return {};

    auto filterbank = create_mel_filterbank();

    std::vector<float> window(n_fft_);
    for (int i = 0; i < n_fft_; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft_));
    }

    float* fft_in = fftwf_alloc_real(n_fft_);
    fftwf_complex* fft_out = fftwf_alloc_complex(n_fft_ / 2 + 1);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft_, fft_in, fft_out, FFTW_ESTIMATE);

    std::vector<float> mel_spectrogram(time_steps * n_mels_, 0.0f);

    for (int t = 0; t < time_steps; t++) {
        int start = t * hop_length_;
        for (int i = 0; i < n_fft_; i++) {
            fft_in[i] = audio[start + i] * window[i];
        }

        fftwf_execute(plan);

        std::vector<float> power(n_fft_ / 2 + 1);
        for (int k = 0; k <= n_fft_ / 2; k++) {
            power[k] = (fft_out[k][0] * fft_out[k][0] + fft_out[k][1] * fft_out[k][1]);
        }

        for (int m = 0; m < n_mels_; m++) {
            float energy = 0.0f;
            for (int k = 0; k <= n_fft_ / 2; k++) energy += filterbank[m][k] * power[k];
            mel_spectrogram[t * n_mels_ + m] = energy;
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(fft_in);
    fftwf_free(fft_out);

    // OpenAI Whisper Log10 compression
    float log_spec_max = -1e10f;
    for (auto& val : mel_spectrogram) {
        val = std::log10(std::max(val, 1e-10f));
        log_spec_max = std::max(log_spec_max, val);
    }
    for (auto& val : mel_spectrogram) {
        val = std::max(val, log_spec_max - 8.0f);
        val = (val + 4.0f) / 4.0f;
    }

    // Transpose [time_steps, n_mels] -> [n_mels, time_steps]
    std::vector<float> transposed(n_mels_ * time_steps);
    for (int t = 0; t < time_steps; t++) {
        for (int m = 0; m < n_mels_; m++) {
            transposed[m * time_steps + t] = mel_spectrogram[t * n_mels_ + m];
        }
    }

    return transposed;
}

void AudioPreprocessor::save_safetensors(
    const std::vector<float>& transposed_mel, int time_steps,
    const std::string& output_path) {
    std::vector<uint16_t> fp16_data(transposed_mel.size());
    for (size_t i = 0; i < transposed_mel.size(); i++) {
        fp16_data[i] = float_to_fp16(transposed_mel[i]);
    }

    size_t data_size = fp16_data.size() * sizeof(uint16_t);

    std::string json_header = "{\"mel_spectrogram\": {\"dtype\": \"F16\", \"shape\": [1, " +
                              std::to_string(n_mels_) + ", " + std::to_string(time_steps) +
                              "], \"data_offsets\": [0, " + std::to_string(data_size) + "]}}";

    size_t header_size = json_header.size();
    size_t padded_size = ((header_size + 7) / 8) * 8;
    json_header.resize(padded_size, ' ');

    std::ofstream st_file(output_path, std::ios::binary);
    uint64_t hlen = padded_size;
    st_file.write(reinterpret_cast<const char*>(&hlen), 8);
    st_file.write(json_header.c_str(), padded_size);
    st_file.write(reinterpret_cast<const char*>(fp16_data.data()), data_size);
}

}  // namespace bot_speech
