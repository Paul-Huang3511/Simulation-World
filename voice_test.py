#!/usr/bin/env python3
"""test_voice.py — 语音识别独立测试"""

import sounddevice as sd
import whisper

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
WHISPER_MODEL = "base"

def main():
    # 1. 加载模型
    print(f"⏳ 加载 Whisper ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    print("✅ Whisper 就绪\n")

    while True:
        user_input = input("👉 按回车开始录音（输入 q 退出）: ").strip()
        if user_input.lower() == "q":
            print("👋 退出")
            break

        # 2. 录音
        print(f"🔴 录音中…（最长 {RECORD_SECONDS} 秒）")
        try:
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()
        except Exception as e:
            print(f"❌ 麦克风出错：{e}")
            continue

        # 3. 识别
        print("🧠 Whisper 识别...")
        result = model.transcribe(audio, language="zh", fp16=False)
        text = result["text"].strip()
        print(f"📝 识别结果: {text}\n")

if __name__ == "__main__":
    main()
