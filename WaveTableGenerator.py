import wave
import struct
import os
import numpy as np
from sf2utils.sf2parse import Sf2File
from io import StringIO
from string import Template
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


def auto_fix_loop_point(
    attack,
    loop,
    search_window=64,
    match_len=8,
    slope_weight=0.5
):
    """
    Automatically adjust loop start point to minimize discontinuity
    after FFT band-limiting.

    attack        : attack samples (1D np.array)
    loop          : loop samples (1D np.array)
    search_window : ± samples to search around original loop start
    match_len     : number of samples to compare
    slope_weight  : weight for first-derivative continuity
    """

    full = np.concatenate((attack, loop))
    attack_len = len(attack)
    loop_len = len(loop)

    best_err = float("inf")
    best_i = attack_len

    start = max(match_len + 1, attack_len - search_window)
    end   = min(len(full) - loop_len - match_len - 1,
                attack_len + search_window)

    for i in range(start, end):
        err = 0.0
        for n in range(match_len):
            a0 = full[i + n]
            b0 = full[i + loop_len + n]

            a1 = full[i + n] - full[i + n - 1]
            b1 = full[i + loop_len + n] - full[i + loop_len + n - 1]

            err += (a0 - b0) ** 2
            err += slope_weight * (a1 - b1) ** 2

        if err < best_err:
            best_err = err
            best_i = i

    new_attack = full[:best_i]
    new_loop   = full[best_i:best_i + loop_len]

    print(f"[Loop Fix] Loop start adjusted by {best_i - attack_len} samples")

    return new_attack.astype(attack.dtype), new_loop.astype(loop.dtype)


def compute_loop_error_curve(
    attack,
    loop,
    search_window=64,
    match_len=8,
    slope_weight=0.5
):
    """
    Compute loop error curve around original loop point.
    Returns offsets and corresponding error values.
    """

    full = np.concatenate((attack, loop))
    attack_len = len(attack)
    loop_len = len(loop)

    offsets = []
    errors = []

    start = max(match_len + 1, attack_len - search_window)
    end   = min(len(full) - loop_len - match_len - 1,
                attack_len + search_window)

    for i in range(start, end):
        err = 0.0
        for n in range(match_len):
            a0 = full[i + n]
            b0 = full[i + loop_len + n]

            a1 = full[i + n] - full[i + n - 1]
            b1 = full[i + loop_len + n] - full[i + loop_len + n - 1]

            err += (a0 - b0) ** 2
            err += slope_weight * (a1 - b1) ** 2

        offsets.append(i - attack_len)
        errors.append(err)

    return np.array(offsets), np.array(errors)

def apply_gain_db(samples, gain_db, sample_width):
    """
    Apply gain in dB to integer samples with clipping.
    samples      : np.array
    gain_db      : float (e.g. -6.0)
    sample_width : 1 or 2 (bytes)
    """
    if gain_db == 0.0:
        return samples

    gain = pow(10.0, gain_db / 20.0)

    samples_f = samples.astype(np.float32) * gain

    if sample_width == 1:
        min_v, max_v = -128, 127
    elif sample_width == 2:
        min_v, max_v = -32768, 32767
    else:
        raise RuntimeError("Unsupported sample width")

    samples_f = np.clip(samples_f, min_v, max_v)

    return samples_f.astype(samples.dtype)


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

def plot_loop_error_curve_to_pdf(
    pdf,
    offsets,
    errors,
    best_offset,
    title
):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(offsets, errors, label="Loop Error", linewidth=1.5)
    ax.axvline(0, color='gray', linestyle='--', label="Original Loop")
    ax.axvline(best_offset, color='r', linestyle='--',
               label=f"Fixed Loop ({best_offset:+d} samples)")

    ax.set_xlabel("Loop Start Offset (samples)")
    ax.set_ylabel("Error (value + slope)")
    ax.set_title(title)
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
        parser.add_argument('--gainDb', type=float, default=0.0,
                    help='Sample gain in dB (e.g. -6.0, default 0 dB)')
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

            samples = bandlimit_by_lowest_note(
                np.concatenate((attackSamples, loopSamples)),
                args.outSampleRate,
                sampleFreqEst,
                lowestFreq
            )

            attackSamples = samples[0:len(attackSamples)]
            loopSamples   = samples[len(attackSamples):]

            # --- Loop error BEFORE fix ---
            offsets_before, errors_before = compute_loop_error_curve(
                attackSamples,
                loopSamples,
                search_window=256,
                match_len=8,
                slope_weight=0.5
            )

            attackFixed, loopFixed = auto_fix_loop_point(
                attackSamples,
                loopSamples,
                search_window=256,
                match_len=8,
                slope_weight=0.5
            )

            offsets_after, errors_after = compute_loop_error_curve(
                attackFixed,
                loopFixed,
                search_window=256,
                match_len=8,
                slope_weight=0.5
            )

            attackSamples = attackFixed
            loopSamples   = loopFixed

            # --- Apply user gain ---
            if args.gainDb != 0.0:
                print(f"Applying sample gain: {args.gainDb:.2f} dB")

                attackSamples = apply_gain_db(
                    attackSamples, args.gainDb, args.outSampleWidth)

                loopSamples = apply_gain_db(
                    loopSamples, args.gainDb, args.outSampleWidth)
                
            pdf = None
            if args.spectrumPdf:
                pdf = PdfPages(args.spectrumPdf)
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

                best_offset = offsets_before[np.argmin(errors_before)]

                plot_loop_error_curve_to_pdf(
                    pdf,
                    offsets_before,
                    errors_before,
                    best_offset,
                    title=f"{sampleName} Loop Error (Before Fix)"
                )

                best_offset_after = offsets_after[np.argmin(errors_after)]

                plot_loop_error_curve_to_pdf(
                    pdf,
                    offsets_after,
                    errors_after,
                    best_offset_after,
                    title=f"{sampleName} Loop Error (After Fix)"
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
