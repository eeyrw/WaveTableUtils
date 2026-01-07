import wave
import struct
import os
from math import e, log2, pow
import numpy as np
from sf2utils.sf2parse import Sf2File
from io import StringIO
from string import Template
import struct
import argparse
import pprint
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def resample(inSamples, inSampleRate, inChannel, outSampleRate):
    targetSampleLen = outSampleRate*len(inSamples)//inSampleRate
    if inChannel == 1:
        outSamples = signal.resample(inSamples, targetSampleLen)
    elif inChannel == 2:
        outSamples = signal.resample(inSamples, targetSampleLen, axis=1)
    else:
        raise RuntimeError(
            'Can not resample samples with channel num =  %d!' % inChannel)
    return outSamples


def chanAdjust(inSamples, inChannel, outChannel):
    if inChannel == outChannel:
        return inSamples
    elif inChannel == 2 and outChannel == 1:
        return np.sum(inSamples, axis=0)//2
    elif inChannel == 1 and outChannel == 2:
        np.stack((inSamples, inSamples))
    else:
        raise RuntimeError(
            'Can not convert channel %d to channel %d!' % (inChannel, outChannel))


def widthAdjust(inSamples, inWidth, outWidth):
    if inWidth == outWidth:
        return inSamples
    elif inWidth == 2 and outWidth == 1:
        return inSamples//256
    elif inWidth == 1 and outWidth == 2:
        return inSamples*256
    else:
        raise RuntimeError(
            'Can not convert width %d to width %d!' % (inWidth, outWidth))


def samplePipelineProcess(inSamples, inChannel, inWidth, inSampleRate, outChannel, outWidth, outSampleRate):
    midOut = resample(inSamples, inSampleRate, inChannel, outSampleRate)
    midOut = chanAdjust(midOut, inChannel, outChannel)
    midOut = widthAdjust(midOut, inWidth, outWidth)
    return midOut


def noteToFreq(note):
    a = 440  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


def readWaveSamples(file_path):
    sampleArray = []
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    if nchannels != 1:
        print("Must be mono wave file.")
        return -1
    if sampwidth != 2:
        print("Must be 16bit sample width.")
        return -1
    if framerate != 32000:
        print("Sample rate must be 32000 samples/s")
        return -1

    for i in range(0, nframes):
        waveData = f.readframes(1)
        data = struct.unpack("<h", waveData)
        sampleArray.append(int(data[0]))
    f.close()
    return sampleArray


def calcIncrement(baseFreq, targetFreq):
    return targetFreq/baseFreq


def estimateSampleFreq(samples, sampleRate):
    estimateFreq = 0
    chunk = len(samples)
    # use a Blackman window
    window = np.blackman(chunk)
    # unpack the data and times by the hamming window
    indata = np.array(samples)*window
    # Take the fft and square each value
    fftData = abs(np.fft.rfft(indata))**2
    # find the maximum
    which = fftData[1:].argmax() + 1
    # use quadratic interpolation around the max
    if which != len(fftData)-1:
        y0, y1, y2 = np.log(fftData[which-1:which+2:])
        x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
        # find the frequency and output it
        estimateFreq = (which+x1)*sampleRate/chunk
    else:
        estimateFreq = which*sampleRate/chunk
    return estimateFreq


def bandlimit_by_lowest_note(
    samples,
    sample_rate,
    base_freq,
    lowest_target_freq,
    transition_ratio=0.15
):
    """
    FFT band-limit samples so that when transposed down to lowest_target_freq,
    no aliasing occurs.

    samples               : 1D numpy array
    sample_rate           : Hz
    base_freq             : original sample fundamental frequency (Hz)
    lowest_target_freq    : lowest playback frequency (Hz)
    transition_ratio      : soft roll-off width
    """

    samples = samples.astype(np.float32)

    N = len(samples)
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(N, 1.0 / sample_rate)

    nyquist = sample_rate * 0.5
    f_max = nyquist * (lowest_target_freq / base_freq)

    f_start = f_max * (1.0 - transition_ratio)
    f_stop  = f_max

    window = np.ones_like(freqs)

    for i, f in enumerate(freqs):
        if f <= f_start:
            window[i] = 1.0
        elif f >= f_stop:
            window[i] = 0.0
        else:
            x = (f - f_start) / (f_stop - f_start)
            window[i] = np.cos(x * np.pi * 0.5) ** 2

    spectrum *= window

    out = np.fft.irfft(spectrum, n=N)

    # normalize back to original range
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= np.max(np.abs(samples)) / peak

    return out.astype(samples.dtype)

def plot_spectrum_to_pdf(
    pdf,
    original,
    processed,
    sample_rate,
    title,
    f_max=None
):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    N = min(len(original), len(processed))
    window = np.blackman(N)

    spec_orig = np.fft.rfft(original[:N] * window)
    spec_proc = np.fft.rfft(processed[:N] * window)

    freqs = np.fft.rfftfreq(N, 1.0 / sample_rate)

    mag_orig = 20 * np.log10(np.abs(spec_orig) + 1e-12)
    mag_proc = 20 * np.log10(np.abs(spec_proc) + 1e-12)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(freqs, mag_orig, label="Original", alpha=0.6)
    ax.plot(freqs, mag_proc, label="Band-limited", alpha=0.85)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if f_max is not None:
        ax.axvline(f_max, color='r', linestyle='--',
                   label="f_max (lowest note)")
        ax.legend()

        ax.set_xlim(0, f_max * 1.5)
    else:
        ax.set_xlim(0, sample_rate / 2)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_time_domain_to_pdf(
    pdf,
    original,
    processed,
    sample_rate,
    title,
    time_ms=10.0
):
    """
    Plot time-domain waveform comparison (original vs processed)
    into PDF.
    """

    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    max_samples = int(sample_rate * time_ms / 1000.0)
    max_samples = min(max_samples, len(original), len(processed))

    t = np.arange(max_samples) / sample_rate * 1000.0  # ms

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(t, original[:max_samples],
            label="Original", alpha=0.6)
    ax.plot(t, processed[:max_samples],
            label="Band-limited", alpha=0.85)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{title} (First {time_ms:.1f} ms)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def getCStyleSampleDataString(sampleArray, colWidth):
    file_str = StringIO()
    newLineCounter = 0
    for sample in sampleArray:
        file_str.write("%6d," % sample)
        if newLineCounter > colWidth:
            newLineCounter = 0
            file_str.write("\n")
        else:
            newLineCounter += 1
    return file_str.getvalue()


def formatFileByParam(templateFile, outputFile, param):
    with open(templateFile, 'r') as tmplFile:
        tmplString = tmplFile.read()
        s = Template(tmplString)
        with open(outputFile, 'w') as outFile:
            outFile.write(s.safe_substitute(param))

# is_left:0
# is_mono:1
# loop_duration:322
# name:'Music Box B4'
# original_pitch:83
# pitch_correction:0
# raw_sample_data:b'\xf2\xff\x05\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x06\x00\xf1\xff\x07\x00\xf3\xff\x08\x00\xf4\xff\x0b\x00\xf5\xff\r\x00\xf8\xff\x10\x00\xfd\xff\x15\x00\x00\x00\x1a\x00\x03\x00\x1a\x00\x07\x00\x1f\x00\x0e\x00#\x00\xeb\xff:\x00}\x00\n\x01\x10\x02w\x03\xad\x04\xfc\x06\xcf\x07\xb7\t%\x0c\x96\rM\x0f4\x12\xd6\x12\xb9\x14v\x16\x1d\x18U\x1a\xbb\x19\xa6\x1b\xf6\x185\x17\xe8\x18\xce\x15\xd6\x123\x11\x00\x10h\x0f\x8b\x0f9\x0e!\x0e\x0e\x0b\xb6\x07\x10\xffh\xf81\xf7l\xf1\xf5\xeb{\xe7a\xe2\xfa\xdd\xb0\xda\x97\xd8\x1b\xd8s\xdb\x19\xdc(\xd9n\xd7\x18\xd7!\xd8\xc9\xd6\xf5\xd1o\xcb\xbb\xc7\xf3\xca\xfc\xcf\xb1\xce\xbb\xd2(\xdc\xfd\xe5;\xf30\xf5@\xfd\x8c\x02\x0c\t\xd9\x0f\x81\x0b\x18\x0e\xb5\x0f\xf0\x12\x93\x15\xfa\x12\xec\x12?\x17F\x1d:#\x88$\xed\'Y346\xab?\xbe;\xd77\xe971-S+\x08\x1do\x14\xe4\x0f$\tL\tz\x07f\x07\xde\x0c]\x11\xd9\x12\xd5\x14z\x11\x99\x12L\x08\n\xfeY\xf4V\xe5\'\xe1\xa7\xd5>\xd0\xe6\xcc\xab\xcb\xe5\xd1\x04\xd2\x...
# sample_link:None
# sample_rate:32000
# sample_type:1
# sample_width:2
# sf2parser:<sf2utils.sf2parse.Sf2File object at 0x000001EA66321128>
# sm24_offset:None
# smpl_offset:214
# start:0
# start_loop:57637


def getFromSf2(sf2FilePath, sampleName):
    with open(sf2FilePath, 'rb') as sf2_file:
        sf2 = Sf2File(sf2_file)

        for sample in sf2.samples:
            if sample.name == sampleName:
                if sample.sample_width == 1:
                    upackStr = '<b'
                elif sample.sample_width == 2:
                    upackStr = '<h'
                else:
                    raise RuntimeError('Unsupported sample width')
                samples = [sampleValue[0] for sampleValue in struct.iter_unpack(
                    upackStr, sample.raw_sample_data)]
                attackSamples = samples[0:sample.start_loop]
                loopSamples = samples[sample.start_loop:sample.end_loop]
                sampleMidiNote = sample.original_pitch
                return (sampleName, sampleMidiNote, np.array(attackSamples),  np.array(loopSamples), sample.sample_width, sample.sample_rate, 1)


def listSf2Info(sf2FilePath):
    with open(sf2FilePath, 'rb') as sf2_file:
        sf2 = Sf2File(sf2_file)

        # We do not care instrument by now.
        # for instrument in sf2.instruments:
        #     if instrument.name != 'EOI':
        #         pprint.pprint(vars(instrument))
        #         for bag in instrument.bags:
        #             pprint.pprint(bag.__dir__())

        for sample in sf2.samples:
            pprint.pprint(sample)


def genCode(templateFiles, sampleName, sampleFreq, sampleRate, sampleWidth, attackSamples, loopSamples, outputDir, padding):
    attackLen = len(attackSamples)
    loopLen = len(loopSamples)
    totalLen = attackLen+loopLen
    if padding:
        paddingLen = 1
        loopSamples = np.append(loopSamples, loopSamples[0])
    else:
        paddingLen = 0
    if sampleWidth == 1:
        sampleType = "int8_t"
    elif sampleWidth == 2:
        sampleType = "int16_t"
    attackSamplesDataString = getCStyleSampleDataString(
        attackSamples, 8)
    loopSamplesDataString = getCStyleSampleDataString(
        loopSamples, 8)
    incrementDataString = getCStyleSampleDataString([calcIncrement(sampleFreq, noteToFreq(
        i))*255 for i in range(0, 128)], 8)
    print("Estimated base frequency:%f Hz" % sampleFreq)
    paramDict = {}
    paramDict['WaveTableName'] = sampleName
    paramDict['WaveTableBaseFreq'] = sampleFreq
    paramDict['WaveTableSampleRate'] = sampleRate
    paramDict['WaveTableLen'] = totalLen
    paramDict['WaveTableActualLen'] = totalLen+paddingLen
    paramDict['WaveTableAttackLen'] = attackLen
    paramDict['WaveTableLoopLen'] = loopLen
    paramDict['WaveTableSampleType'] = sampleType
    paramDict['WaveTableIncrementType'] = "uint16_t"
    paramDict['WaveTableAttackData'] = attackSamplesDataString
    paramDict['WaveTableLoopData'] = loopSamplesDataString
    paramDict['WaveTableIncrementData'] = incrementDataString

    for templateFile in templateFiles:
        formatFileByParam(templateFile, os.path.join(outputDir, os.path.basename(os.path.splitext(
            templateFile)[0])), paramDict)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='The wavetable c style code generation.')
        parser.add_argument('--template', type=str, default='generic',
                            help='Using interal template by specifing type.')
        parser.add_argument('--sf2', type=str, default='',
                            help='Sondfont2 file path.')
        parser.add_argument('--listSf2', default=False, action='store_true',
                            help='List infomation of sf2 file.')
        parser.add_argument('--sampleName', type=str, default='Celesta C5 Mini',
                            help='Wavetable sample name.')
        parser.add_argument('--outSampleRate', type=int, default=32000,
                            help='Output wavetable sample rate.')
        parser.add_argument('--outSampleWidth', type=int, default=1,
                            help='Wavetable sample wdith.')
        parser.add_argument('--lowestNote', type=int, default=36,
                    help='Lowest MIDI note to be played (for FFT band-limit).')
        parser.add_argument('--spectrumPdf', type=str, default='',
                    help='Output spectrum comparison chart to PDF file.')
        parser.add_argument('--padding', default=False, action='store_true',
                            help='Padding one sample at the end of table in aspect of convenience of interpolation.')
        parser.add_argument('--extraTemplate', nargs='+', type=str, default=[],
                            help='Using extra template files instead of self-contained template.')
        parser.add_argument('--outputDir', type=str, default='.',
                            help='Output directory.')
        args = parser.parse_args()
        if args.template != None:
            templateFileList = []
            for filePath in os.listdir(os.path.join('./template', args.template)):
                if os.path.splitext(filePath)[1] == '.template':
                    templateFileList.append(os.path.join(
                        './template', args.template, filePath))
        else:
            templateFileList = args.extraTemplate

        if args.sf2 != '' and not args.listSf2:
            (sampleName, sampleMidiNote, attackSamples, loopSamples,
             sampleWidth, sampleRate, sampleChannels) = getFromSf2(args.sf2, args.sampleName)
            attackSamples = samplePipelineProcess(
                attackSamples, sampleChannels, sampleWidth, sampleRate, 1, args.outSampleWidth, args.outSampleRate)
            loopSamples = samplePipelineProcess(
                loopSamples, sampleChannels, sampleWidth, sampleRate, 1, args.outSampleWidth, args.outSampleRate)
            sampleFreqEst = estimateSampleFreq(
                np.concatenate((attackSamples, loopSamples)), args.outSampleRate)

            lowestFreq = noteToFreq(args.lowestNote)

            print("Applying FFT band-limit for lowest freq: %.2f Hz" % lowestFreq)

            attackOrig = attackSamples.copy()
            loopOrig   = loopSamples.copy()

            np.concatenate((attackSamples, loopSamples))

            samples = bandlimit_by_lowest_note(
                np.concatenate((attackSamples, loopSamples)),
                args.outSampleRate,
                sampleFreqEst,
                lowestFreq
            )

            attackSamples = samples[0:len(attackSamples)]
            loopSamples   = samples[len(attackSamples):]


            pdf = None
            if args.spectrumPdf:
                pdf = PdfPages(args.spectrumPdf)
                if pdf is not None:
                    nyquist = args.outSampleRate * 0.5
                    f_max = nyquist * (lowestFreq / sampleFreqEst)
                plot_spectrum_to_pdf(
                    pdf,
                    np.concatenate((attackOrig, loopOrig)),
                    samples,
                    args.outSampleRate,
                    title=f"{sampleName} (Attack + Loop) Spectrum",
                    f_max=f_max
                )

               

                plot_time_domain_to_pdf(
                    pdf,
                    np.concatenate((attackOrig, loopOrig)),
                    samples,
                    args.outSampleRate,
                    title=f"{sampleName} (Attack + Loop) Time Domain",
                    time_ms=1000.0
                )


            sampleFreqFromSf2 = noteToFreq(sampleMidiNote)

            if abs(sampleFreqFromSf2-sampleFreqEst) > 10:
                print('Big diff between sample freq:%.3f and sample est freq:%.3f' % (
                    sampleFreqFromSf2, sampleFreqEst))

            genCode(templateFileList, sampleName, sampleFreqEst, args.outSampleRate,
                    args.outSampleWidth, attackSamples, loopSamples, args.outputDir, args.padding)
        elif args.sf2 != '' and args.listSf2:
            listSf2Info(args.sf2)
        else:
            pass
        if pdf is not None:
            pdf.close()
            print("Spectrum PDF saved to:", args.spectrumPdf)

    except RuntimeError as identifier:
        print('Meet error during code generation: '+str(identifier))
