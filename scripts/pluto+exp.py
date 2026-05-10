import adi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import spectrogram

sdr = adi.ad9361(uri="ip:192.168.2.1")

sdr.sample_rate = int(5e6)
sdr.rx_rf_bandwidth = int(5e6)
sdr.rx_buffer_size = 16384
sdr.rx_enabled_channels = [0, 1]

# 처음 몇 번은 버림
for _ in range(3):
    _ = sdr.rx()

out_dir = Path("outputs/figures/pluto_spectrogram_test")
out_dir.mkdir(parents=True, exist_ok=True)

center_freqs = [2.412e9, 2.437e9, 2.462e9]

for cf in center_freqs:
    sdr.rx_lo = int(cf)

    # LO 바꾼 직후 안정화용으로 한 번 버림
    _ = sdr.rx()

    data = sdr.rx()
    ch0 = data[0]

    f, t, Sxx = spectrogram(
        ch0,
        fs=int(sdr.sample_rate),
        nperseg=512,
        noverlap=384,
        return_onesided=False,
        mode="magnitude",
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f / 1e6, Sxx_db, shading="gouraud")
    plt.title(f"RX0 Spectrogram @ {cf/1e9:.3f} GHz")
    plt.ylabel("Baseband Frequency [MHz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Magnitude [dB]")
    plt.tight_layout()

    out = out_dir / f"spectrogram_{int(cf/1e6)}MHz.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"saved: {out}")