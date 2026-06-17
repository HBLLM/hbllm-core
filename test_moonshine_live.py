#!/usr/bin/env python3
"""Test Moonshine ONNX directly with a generated speech-like signal and a real recording."""
import numpy as np
import sys

print("=== Moonshine Live Debug ===")

# 1. Test with a known signal (sine wave at 440Hz — should return empty or noise)
print("\n--- Test 1: 440Hz sine wave (1s) ---")
sr = 16000
t = np.linspace(0, 1, sr, dtype=np.float32)
sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

model = MoonshineOnnxModel(model_name="moonshine/base")
tokenizer = load_tokenizer()

tokens = model.generate(sine_wave.reshape(1, -1))
print(f"  Token IDs: {tokens}")
texts = tokenizer.decode_batch(tokens)
print(f"  Decoded: '{texts[0] if texts else ''}'")

# 2. Test with random noise (should return empty)
print("\n--- Test 2: Random noise (1s) ---")
noise = np.random.randn(sr).astype(np.float32) * 0.3
tokens = model.generate(noise.reshape(1, -1))
print(f"  Token IDs: {tokens}")
texts = tokenizer.decode_batch(tokens)
print(f"  Decoded: '{texts[0] if texts else ''}'")

# 3. Test with the moonshine_onnx.transcribe function on a WAV file
print("\n--- Test 3: Record 3s from mic and transcribe ---")
try:
    import sounddevice as sd
    print("  Recording 3 seconds... SPEAK NOW!")
    audio = sd.rec(int(3 * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"  Audio stats: rms={rms:.4f}, peak={peak:.4f}, len={len(audio)} ({len(audio)/sr:.2f}s)")
    
    # Normalize
    if peak > 0.001:
        audio_norm = audio * (0.95 / peak)
        rms2 = float(np.sqrt(np.mean(audio_norm ** 2)))
        print(f"  After norm: rms={rms2:.4f}, peak=0.95")
    else:
        audio_norm = audio
        print("  Audio too quiet to normalize!")
    
    # Test raw
    print("\n  Raw audio:")
    tokens = model.generate(audio.reshape(1, -1))
    print(f"    Token IDs: {tokens}")
    texts = tokenizer.decode_batch(tokens)
    print(f"    Decoded: '{texts[0] if texts else ''}'")
    
    # Test normalized
    print("\n  Normalized audio:")
    tokens = model.generate(audio_norm.reshape(1, -1))
    print(f"    Token IDs: {tokens}")
    texts = tokenizer.decode_batch(tokens)
    print(f"    Decoded: '{texts[0] if texts else ''}'")
    
    # Save to file for inspection
    import soundfile as sf
    sf.write("/tmp/moonshine_test.wav", audio, sr)
    sf.write("/tmp/moonshine_test_norm.wav", audio_norm, sr)
    print("\n  Saved: /tmp/moonshine_test.wav and /tmp/moonshine_test_norm.wav")

except ImportError:
    print("  sounddevice not installed, skipping mic test")
    print("  Install with: pip install sounddevice")

# 4. Test with the higher-level transcribe API
print("\n--- Test 4: Using moonshine_onnx.transcribe() on WAV ---")
try:
    from moonshine_onnx import transcribe
    result = transcribe("/tmp/moonshine_test.wav", model="moonshine/base")
    print(f"  transcribe() result: {result}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Done ===")
