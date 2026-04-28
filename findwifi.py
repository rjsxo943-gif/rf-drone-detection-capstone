import adi
import numpy as np
import matplotlib.pyplot as plt
import time

# Pluto+ 연결
sdr = adi.ad9361("ip:192.168.2.1")

# 기본 설정
sdr.sample_rate = int(20e6)
sdr.rx_rf_bandwidth = int(20e6)
sdr.rx_buffer_size = 8192
sdr.rx_enabled_channels = [0]   # 일단 RX0만 사용

# 2.4GHz Wi-Fi 주요 채널 중심 주파수 (1~13채널)
wifi_channels = {
    1: 2.412e9,
    2: 2.417e9,
    3: 2.422e9,
    4: 2.427e9,
    5: 2.432e9,
    6: 2.437e9,
    7: 2.442e9,
    8: 2.447e9,
    9: 2.452e9,
    10: 2.457e9,
    11: 2.462e9,
    12: 2.467e9,
    13: 2.472e9,
}

powers_db = []
channels = []

for ch, freq in wifi_channels.items():
    sdr.rx_lo = int(freq)

    # 주파수 바꾼 직후 안정화 시간 조금 주기
    time.sleep(0.1)

    data = sdr.rx()
    x = data if isinstance(data, np.ndarray) else data[0]

    # 평균 전력 계산
    power = np.mean(np.abs(x) ** 2)
    power_db = 10 * np.log10(power + 1e-12)

    channels.append(ch)
    powers_db.append(power_db)

    print(f"Channel {ch:2d} | Center {freq/1e9:.3f} GHz | Power = {power_db:.2f} dB")

# 가장 강한 채널 찾기
best_idx = int(np.argmax(powers_db))
best_ch = channels[best_idx]
best_power = powers_db[best_idx]

print("\n=== Scan Result ===")
print(f"Strongest channel: {best_ch}")
print(f"Power: {best_power:.2f} dB")

# 그래프 저장
plt.figure(figsize=(8, 4))
plt.plot(channels, powers_db, marker='o')
plt.title("2.4 GHz Wi-Fi Channel Scan")
plt.xlabel("Wi-Fi Channel")
plt.ylabel("Average Power [dB]")
plt.xticks(channels)
plt.grid(True)
plt.tight_layout()
plt.savefig("wifi_channel_scan.png")
plt.close()

print("saved: wifi_channel_scan.png")

del sdr