import math
import matplotlib.pyplot as plt

AUDIO_BITS = 10
PWM_BITS = 16

AUDIO_MAX = (1 << AUDIO_BITS) - 1     # 1023
PWM_MAX   = (1 << PWM_BITS) - 1       # 65535

K = 100.0   # 对数曲线强度（越大，低幅度越亮）

lut = []

for i in range(AUDIO_MAX + 1):
    x = i / AUDIO_MAX
    y = math.log1p(K * x) / math.log1p(K)
    pwm = int(y * PWM_MAX + 0.5)
    lut.append(pwm)

# ---- 打印为 C 数组（可直接拷进 MCU）----
print("const uint16_t audio2pwm_lut[1024] = {")
for i in range(0, 1024, 8):
    print("    " + ", ".join(f"{v:5d}" for v in lut[i:i+8]) + ",")
print("};")

# ---- 可视化曲线 ----
plt.plot(lut)
plt.title("10-bit Audio → 16-bit PWM (Log Mapping)")
plt.xlabel("Audio amplitude (0..1023)")
plt.ylabel("PWM value (0..65535)")
plt.grid(True)
plt.show()
