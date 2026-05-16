#ifndef BOT_SPEECH__AUDIO_PREPROCESSOR_HPP_
#define BOT_SPEECH__AUDIO_PREPROCESSOR_HPP_

#include <vector>
#include <string>
#include <cstdint>

namespace bot_speech
{

class AudioPreprocessor
{
public:
  AudioPreprocessor();

  /**
   * @brief 加载 WAV 文件
   * @param filepath WAV 文件路径
   * @return float32 音频数据
   */
  std::vector<float> load_wav(const std::string & filepath);

  /**
   * @brief 提取 Whisper Mel-spectrogram
   * @param audio 音频数据
   * @param out_time_steps 输出时间步数
   * @return 转置后的 Mel-spectrogram [n_mels, time_steps]
   */
  std::vector<float> extract_whisper_mel(
    const std::vector<float> & audio, int & out_time_steps);

  /**
   * @brief 保存为 safetensors 格式
   * @param transposed_mel 转置后的 Mel-spectrogram
   * @param time_steps 时间步数
   * @param output_path 输出路径
   */
  void save_safetensors(
    const std::vector<float> & transposed_mel,
    int time_steps,
    const std::string & output_path);

private:
  float hz_to_mel(float f);
  float mel_to_hz(float mel);
  std::vector<std::vector<float>> create_mel_filterbank();
  uint16_t float_to_fp16(float f);

  int sample_rate_ = 16000;
  int n_fft_ = 400;
  int hop_length_ = 160;
  int n_mels_ = 128;
  float f_min_ = 0.0f;
  float f_max_ = 8000.0f;
};

}  // namespace bot_speech

#endif  // BOT_SPEECH__AUDIO_PREPROCESSOR_HPP_
