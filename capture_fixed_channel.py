import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import adi

# =========================
# 사용자 설정
# =========================
URI = "ip:192.168.2.1"   # USB로 잘 잡히면 보통 이거
CENTER_FREQ = 2.472e9    # 채널 13 중심주파수
SAMPLE_RATE = 10e6       # 처음엔 10 MHz로 가볍게 시작
RF_BW = 10e6
RX_GAIN = 40             # 수동 gain
BUF_LEN = 8192           # 한 번에 읽을 샘플 수
NUM_BLOCKS = 200         # 총 수집 블록 수 (8192*200 ≈ 163만 샘플)

# STFT 설정
NPERSEG = 256
NOVERLAP = 192

# =========================
# SDR 설정
# =========================
sdr = adi.Pluto(URI)

sdr.sample_rate = int(SAMPLE_RATE)
sdr.rx_lo = int(CENTER_FREQ)
sdr.rx_rf_bandwidth = int(RF_BW)
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = RX_GAIN
sdr.rx_buffer_size = BUF_LEN
sdr.rx_enabled_channels = [0]

print("=== SDR CONFIG ===")
print(f"URI          : {URI}")
print(f"Center Freq  : {CENTER_FREQ/1e9:.6f} GHz")
print(f"Sample Rate  : {SAMPLE_RATE/1e6:.2f} MHz")
print(f"RF BW        : {RF_BW/1e6:.2f} MHz")
print(f"RX Gain      : {RX_GAIN} dB")
print(f"Buffer Len   : {BUF_LEN}")
print(f"Num Blocks   : {NUM_BLOCKS}")

# =========================
# IQ 수집
# =========================
iq_blocks = []

print("\nCapturing IQ...")
for i in range(NUM_BLOCKS):
    data = sdr.rx()
    data = np.asarray(data).astype(np.complex64)
    iq_blocks.append(data)

    if (i + 1) % 20 == 0:
        print(f"  captured {i+1}/{NUM_BLOCKS} blocks")

iq = np.concatenate(iq_blocks)

print("\n=== CAPTURE DONE ===")
print(f"Total samples: {len(iq)}")
print(f"Duration     : {len(iq)/SAMPLE_RATE:.3f} sec")

# 저장
np.save("iq_ch13.npy", iq)
print("saved: iq_ch13.npy")

# =========================
# PSD 계산
# =========================
window = np.hanning(len(iq))
iq_win = iq * window

fft_data = np.fft.fftshift(np.fft.fft(iq_win))
psd = 20 * np.log10(np.abs(fft_data) + 1e-12)

freq_axis = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1/SAMPLE_RATE))
freq_axis_mhz = freq_axis / 1e6

plt.figure(figsize=(10, 5))
plt.plot(freq_axis_mhz, psd)
plt.xlabel("Frequency Offset (MHz)")
plt.ylabel("Magnitude (dB)")
plt.title("PSD around 2.472 GHz (Channel 13)")
plt.grid(True)
plt.tight_layout()
plt.savefig("psd_ch13.png", dpi=150)
plt.show()
print("saved: psd_ch13.png")

# =========================
# STFT / Spectrogram
# =========================
f, t, Sxx = spectrogram(
    iq,
    fs=SAMPLE_RATE,
    window="hann",
    nperseg=NPERSEG,
    noverlap=NOVERLAP,
    nfft=NPERSEG,
    detrend=False,
    return_onesided=False,
    scaling="density",
    mode="magnitude"
)

# 중심 정렬
Sxx = np.fft.fftshift(Sxx, axes=0)
f = np.fft.fftshift(f)
f_mhz = f / 1e6

Sxx_dB = 20 * np.log10(Sxx + 1e-12)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f_mhz, Sxx_dB, shading="gouraud")
plt.xlabel("Time (s)")
plt.ylabel("Frequency Offset (MHz)")
plt.title("Spectrogram around 2.472 GHz (Channel 13)")
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.savefig("spectrogram_ch13.png", dpi=150)
plt.show()
print("saved: spectrogram_ch13.png")

# =========================
# 간단 통계
# =========================
power = np.mean(np.abs(iq) ** 2)
power_db = 10 * np.log10(power + 1e-12)

print("\n=== SIMPLE STATS ===")
print(f"Mean Power (linear): {power:.6f}")
print(f"Mean Power (dB)    : {power_db:.2f} dB")
print(f"IQ first 10        : {iq[:10]}")