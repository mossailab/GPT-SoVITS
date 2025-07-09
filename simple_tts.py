
import torch
import torchaudio
import argparse
import numpy as np
import os
from inference_webui import get_tts_wav

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(ref_audio_path, ref_text, gen_text, output_path):
    print(">>> 开始生成语音 >>>")
    tts_gen = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language="中文",
        text=gen_text,
        text_language="中文",
        how_to_cut="不切",
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
        speed=1.0,
        if_freeze=False,
        inp_refs=None,
        sample_steps=8,
        if_sr=False,
        pause_second=0.3,
    )
    sr, audio_int16 = next(tts_gen)
    print(f">>> 保存音频到: {output_path}")
    torchaudio.save(output_path, torch.tensor(audio_int16).unsqueeze(0).to(torch.int16), sample_rate=sr)
    print(">>> 完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", type=str, required=True, help="参考音频路径")
    parser.add_argument("--ref_text", type=str, required=True, help="参考音频对应文本")
    parser.add_argument("--text", type=str, required=True, help="要合成的目标文本")
    parser.add_argument("--output", type=str, default="output/tts_result.wav", help="输出音频路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args.ref_audio, args.ref_text, args.text, args.output)
