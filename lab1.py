"""
Лаба 1. Видалення шуму за його шаблоном.

Запуск:
    python lab1.py --signal music.wav --noise beep.wav
Якщо файли не вказані, генеруються тестові сигнали (для перевірки роботи програми).
Залежності: numpy, scipy, matplotlib, soundfile
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import fftconvolve, find_peaks

rng = np.random.default_rng(42)


# ---------- 1. Завантаження ----------
def load_mono(path):
    x, fs = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def synth_signal(fs, dur=8.0):
    """Тестовий 'корисний' сигнал: кілька тонів з модуляцією."""
    t = np.arange(int(fs * dur)) / fs
    x = (0.5 * np.sin(2 * np.pi * 220 * t)
         + 0.3 * np.sin(2 * np.pi * 440 * t + 0.5 * np.sin(2 * np.pi * 2 * t))
         + 0.2 * np.sin(2 * np.pi * 660 * t))
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
    return x


def synth_noise(fs, dur=0.3):
    """Тестовий специфічний звук: частотно-модульований 'писк'."""
    t = np.arange(int(fs * dur)) / fs
    f = 1500 + 2500 * t / dur
    return np.sin(2 * np.pi * np.cumsum(f) / fs) * np.hanning(len(t))


# ---------- 2. Шаблон шуму ----------
def make_template(noise, fs, gauss_sigma=0.02):
    n_min = int(0.2 * fs)  # >= 200 мс
    if len(noise) < n_min:
        raise ValueError("Шаблон шуму коротший за 200 мс")
    noise = noise / np.max(np.abs(noise))
    return noise + rng.normal(0, gauss_sigma, size=len(noise))  # гаусівський шум


# ---------- 3. Додавання шаблону ----------
def add_template(x, tpl, n_occ=4, amp_range=(0.3, 1.0)):
    L = len(tpl)
    # рівномірно розкидані, неперекривні позиції
    slots = np.linspace(0, len(x) - L, n_occ + 2, dtype=int)[1:-1]
    jitter = rng.integers(-L // 4, L // 4, size=n_occ)
    pos = np.clip(slots + jitter, 0, len(x) - L)
    amps = rng.uniform(*amp_range, size=n_occ)  # різна амплітуда
    y = x.copy()
    for p, a in zip(pos, amps):
        y[p:p + L] += a * tpl
    return y, pos, amps


# ---------- 4. Пошук шаблону (нормована крос-кореляція) ----------
def ncc(y, tpl):
    L = len(tpl)
    num = fftconvolve(y, tpl[::-1], mode="valid")
    cs = np.concatenate(([0], np.cumsum(y ** 2)))
    local_energy = cs[L:] - cs[:-L]
    den = np.sqrt(local_energy) * np.linalg.norm(tpl) + 1e-12
    return num / den


def locate(y, tpl, thr=0.3):
    c = ncc(y, tpl)
    peaks, _ = find_peaks(c, height=thr, distance=len(tpl))
    return peaks, c


# ---------- 5. Видалення (МНК, псевдоінверсія Мура-Пенроуза) ----------
def remove_noise(y, tpl, positions):
    out = y.copy()
    A = tpl.reshape(-1, 1)            # (L x 1)
    A_pinv = np.linalg.pinv(A)        # (1 x L)
    alphas = []
    for p in positions:
        seg = y[p:p + len(tpl)]
        alpha = (A_pinv @ seg).item()  # alpha = (A^T A)^-1 A^T seg
        out[p:p + len(tpl)] = seg - alpha * tpl
        alphas.append(alpha)
    return out, np.array(alphas)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal")
    ap.add_argument("--noise")
    ap.add_argument("--occ", type=int, default=4)
    a = ap.parse_args()

    # 1
    if a.signal:
        x, fs = load_mono(a.signal)
    else:
        fs = 44100
        x = synth_signal(fs)
    assert len(x) / fs >= 5, "Аудіо має бути не коротшим за 5 с"
    x = x / np.max(np.abs(x)) * 0.7

    # 2
    if a.noise:
        nz, fs_n = load_mono(a.noise)
        if fs_n != fs:  # проста передискретизація
            from scipy.signal import resample
            nz = resample(nz, int(len(nz) * fs / fs_n))
    else:
        nz = synth_noise(fs)
    tpl = make_template(nz, fs)

    # 3
    y, true_pos, true_amp = add_template(x, tpl, n_occ=a.occ)

    # 4
    found, c = locate(y, tpl)

    # 5
    x_clean, alphas = remove_noise(y, tpl, found)

    # 6
    print(f"fs = {fs} Гц, тривалість = {len(x)/fs:.2f} с, шаблон = {len(tpl)/fs*1000:.0f} мс")
    print("Істинні позиції :", true_pos.tolist())
    print("Знайдені позиції:", found.tolist())
    print("Істинні амплітуди :", np.round(true_amp, 3).tolist())
    print("Оцінені амплітуди :", np.round(alphas, 3).tolist())
    print(f"RMSE(зашумлений, оригінал) = {rmse(y, x):.5f}")
    print(f"RMSE(очищений,  оригінал) = {rmse(x_clean, x):.5f}")

    sf.write("signal_original.wav", x, fs)
    sf.write("signal_noisy.wav", y, fs)
    sf.write("signal_cleaned.wav", x_clean, fs)

    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    ax[0].plot(t, x, lw=0.5); ax[0].set_title("Оригінальний корисний сигнал")
    ax[1].plot(t, y, lw=0.5)
    for p in found:
        ax[1].axvspan(p / fs, (p + len(tpl)) / fs, color="r", alpha=0.25)
    ax[1].set_title("Сигнал з шумом (червоним — знайдені входження шаблону)")
    ax[2].plot(np.arange(len(c)) / fs, c, lw=0.5)
    ax[2].plot(found / fs, c[found], "rx")
    ax[2].set_title("Нормована крос-кореляція")
    ax[3].plot(t, x, lw=0.8, label="оригінал")
    ax[3].plot(t, x_clean, lw=0.5, alpha=0.8, label="очищений")
    ax[3].legend(); ax[3].set_title("Оригінал vs очищений сигнал")
    for axi in ax:
        axi.set_xlabel("t, с")
    plt.tight_layout()
    plt.savefig("result.png", dpi=130)

    # детальне порівняння навколо першого входження
    if len(found):
        p = found[0]; L = len(tpl)
        s, e = max(0, p - L // 2), min(len(x), p + L + L // 2)
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(s, e) / fs, x[s:e], label="оригінал", lw=1.2)
        plt.plot(np.arange(s, e) / fs, y[s:e], label="з шумом", lw=0.7, alpha=0.6)
        plt.plot(np.arange(s, e) / fs, x_clean[s:e], "--", label="очищений", lw=0.9)
        plt.legend(); plt.title("Збільшений фрагмент (1-ше входження)")
        plt.xlabel("t, с"); plt.tight_layout()
        plt.savefig("result_zoom.png", dpi=130)


if __name__ == "__main__":
    main()
