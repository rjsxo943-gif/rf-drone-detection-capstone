import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

sdr = adi.ad9361("ip:192.168.2.1")

sdr.sample_rate = int(20e6)
sdr.rx_rf_bandwidth = int(20e6)
sdr.rx_buffer_size = 16384
sdr.rx_enabled_channels = [0, 1]

center_freqs = [2.412e9, 2.437e9, 2.462e9]

for cf in center_freqs:
    sdr.rx_lo = int(cf)
    data = sdr.rx()
    ch0 = data[0]

    f, t, Sxx = spectrogram(
        ch0,
        fs=sdr.sample_rate,
        nperseg=512,
        noverlap=256,
        return_onesided=False,
        mode="magnitude"
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f / 1e6, Sxx_db, shading="gouraud")
    plt.title(f"RX0 Spectrogram @ {cf/1e9:.3f} GHz")
    plt.ylabel("Frequency [MHz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(f"spectrogram_{int(cf/1e6)}MHz.png")
    plt.close()

    print(f"saved: spectrogram_{int(cf/1e6)}MHz.png")