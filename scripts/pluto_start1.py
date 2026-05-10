import adi
import numpy as np

sdr = adi.ad9361(uri="ip:192.168.2.1")

sdr.sample_rate = int(5e6)
sdr.rx_lo = int(2.4e9)

sdr.rx_enabled_channels = [0, 1]

samples = sdr.rx()

print(type(samples))
print(len(samples))
print(samples[0][:10])
print(samples[1][:10])