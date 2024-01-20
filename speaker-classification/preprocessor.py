import numpy as np
import librosa


def mel_spectrogram(wavfile_path, frame_size=1024, n_mels=128, chunk_seconds=None):
    sig, fs = librosa.load(wavfile_path, sr=16000)

    # normalize between [-1,1]
    sig /= np.max(np.abs(sig), axis=0)

    if not chunk_seconds:
        chunk_samples = len(sig)
    else:
        chunk_samples = fs * chunk_seconds
    mel_spec_all = []
    for i in range(0, len(sig), chunk_samples):
        if len(sig) - i < chunk_samples:
            break
        melspec = librosa.feature.melspectrogram(y=sig[i:i + chunk_samples],
                                                 sr=fs,
                                                 center=True,
                                                 n_fft=frame_size,
                                                 hop_length=int(frame_size / 2),
                                                 n_mels=n_mels)
        melspec = librosa.power_to_db(melspec, ref=1.0)
        mel_spec_all.append(melspec)

    return mel_spec_all
