"""Quick test: extract geometric features from one audio file."""
import traceback
import os

try:
    from src.data.audio import load_audio, extract_window
    from src.data.geometric_features import extract_geometric_features
    import torchaudio

    # Find a real audio file
    for species in os.listdir("data/train_audio"):
        species_dir = os.path.join("data/train_audio", species)
        if os.path.isdir(species_dir):
            files = os.listdir(species_dir)
            if files:
                path = os.path.join(species_dir, files[0])
                break

    print(f"Testing: {path}")

    waveform = load_audio(path)
    print(f"Loaded: {waveform.shape}")

    window = extract_window(waveform, offset=0)
    print(f"Window: {window.shape}")

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
    db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    spec = db(mel(window.unsqueeze(0)))
    print(f"Spec: {spec.shape}")

    spec_np = spec.numpy()
    if spec_np.ndim == 3:
        spec_np = spec_np[0]  # remove batch dim
    print(f"Spec for features: {spec_np.shape}")

    features = extract_geometric_features(
        waveform=window.numpy(),
        spectrogram=spec_np,
        n_bands=16, tda_delay=10, tda_dim=3, tda_max_points=500)
    print(f"Features: {features.shape}")
    print("SUCCESS")

except Exception:
    traceback.print_exc()
