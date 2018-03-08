import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import utils.kaldi.io as kio

ATTRIBUTE_CLS = ['fricative', 'glide', 'nasal', 'oth', 'sil', 'stop', 'voiced', 'vowel']

filename = './data/GEcall-101-G.story-bt.wav'
sample_rate, samples = wavfile.read(filename)


def log_spectrogram(audio, sample_rate, window_size=20,
                    step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def load_scores(fname):
    sc = []
    with open(fname) as f:
        for ln in f:
            ln = ln.strip()
            dig = [float(i) for i in ln.split()]
            sc.append(dig)
    return np.array(sc)


def plot_phone_alignment(ax, phone_ali, lim=3):
    for sep in phone_ali:
        xs = int(sep[0]) / 1e7
        xe = int(sep[1]) / 1e7
        d = 300000. / 1e7
        if xe < lim:
            ax.axvline(x=xe, c='r', ls='--', lw=0.5)
            if not(sep[2] == 'pau'):
                mid = (xe + xs - d)/2.
                ax.text(mid, 500, sep[2], fontsize=10)


freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
# number of frames to plot
total_time = len(samples) / float(sample_rate)
N_spec = 300
N_sig = N_spec * sample_rate * total_time / len(spectrogram) # number of audio sample
length = N_sig / sample_rate

signal_time = np.linspace(0, length, N_sig)
spec_time = times[:N_spec]

# attribute probabilities
Y_prob = load_scores('./data/GEcall-101-G_res.prob')
Y_prob = Y_prob[:N_spec, :]
# phone alignments
ali = kio.read_mlf('./data/GEcall-101-G_phone.mlf')

fig, axs = plt.subplots(8, 1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.5)
axs = axs.ravel()

# audio signal
axs[0].plot(signal_time, samples[:N_sig])
axs[0].set_xlim([0, signal_time[-1]])
axs[0].set_xticks([], [])
# spectrogram
axs[1].imshow(spectrogram.T[:, :N_spec], aspect='auto', origin='lower', interpolation='nearest',
           extent=[spec_time.min(), spec_time.max(), freqs.min(), freqs.max()])
axs[1].set_yticks(freqs[::60])
axs[1].set_xticks(spec_time[::30])
axs[1].set_ylabel('Hz')
axs[1].set_xticks([], [])

plot_phone_alignment(axs[1], phone_ali=ali['"*/GEcall-101-G.story-bt.lab"'])

cnt = 2
for i, lab in enumerate(ATTRIBUTE_CLS):
    if (lab is not 'oth') and (lab is not 'sil'):
        axs[cnt].plot(spec_time, Y_prob[:, i])
        axs[cnt].set_ylabel(lab, rotation=60)
        axs[cnt].set_xlim([spec_time[0], spec_time[-1]])
        axs[cnt].set_ylim([0, 1.1])
        axs[cnt].set_xticks([], [])
        plot_phone_alignment(axs[cnt], phone_ali=ali['"*/GEcall-101-G.story-bt.lab"'])
        cnt += 1
axs[cnt-1].set_xlim([spec_time[0], spec_time[-1]])
axs[cnt-1].set_xticks(spec_time[::16])
axs[cnt-1].set_xlabel('Seconds')
plt.show()


# fig = plt.figure(figsize=(8, 5))
# # audio signal
# ax1 = fig.add_subplot(211)
# ax1.plot(signal_time, samples[:N_sig])
# ax1.set_title('Raw wave of ' + filename)
# ax1.set_ylabel('Amplitude')
# # spectrogram
# ax2 = fig.add_subplot(212)
# ax2.imshow(spectrogram.T[:, :N_spec], aspect='auto', origin='lower',
#            extent=[spec_time.min(), spec_time.max(), freqs.min(), freqs.max()])
# ax2.set_yticks(freqs[::16])
# ax2.set_xticks(spec_time[::30])
# ax2.set_title('Spectrogram of ' + filename)
# ax2.set_ylabel('Freqs in Hz')
# ax2.set_xlabel('Seconds')
# plt.show()
