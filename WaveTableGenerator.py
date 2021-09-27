import wave
import struct
import os
from math import log2, pow
import numpy as np
from sf2utils.sf2parse import Sf2File
from io import StringIO
from string import Template
import struct
import argparse
import pprint
from scipy import signal


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


def getCStyleSampleDataString(sampleArray, colWidth, dataDescription=''):
    file_str = StringIO()
    newLineCounter = 0
    file_str.write(dataDescription+'\n')
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

        # We do not care instrument by now.
        # for instrument in sf2.instruments:
        #     if instrument.name != 'EOI':
        #         pprint.pprint(vars(instrument))
        #         for bag in instrument.bags:
        #             pprint.pprint(bag.__dir__())

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
                return (sampleName, np.array(attackSamples),  np.array(loopSamples), sample.sample_width, sample.sample_rate, 1)


def genCode(templateFiles, sampleName, sampleFreq, sampleRate, sampleWidth, attackSamples, loopSamples, outputDir):
    attackLen = len(attackSamples)
    loopLen = len(loopSamples)
    totalLen = attackLen+loopLen
    if sampleWidth == 1:
        sampleType = "int8_t"
    elif sampleWidth == 2:
        sampleType = "int16_t"
    attackSamplesDataString = getCStyleSampleDataString(
        attackSamples, 8, dataDescription='// Attack Samples:')
    loopSamplesDataString = getCStyleSampleDataString(
        loopSamples, 8, dataDescription='// Loop Samples:')
    incrementDataString = getCStyleSampleDataString([calcIncrement(sampleFreq, noteToFreq(
        i))*255 for i in range(0, 128)], 8, dataDescription='// Increment:')
    print("Estimated base frequency:%f Hz" % sampleFreq)
    paramDict = {}
    paramDict['WaveTableName'] = sampleName
    paramDict['WaveTableBaseFreq'] = sampleFreq
    paramDict['WaveTableSampleRate'] = sampleRate
    paramDict['WaveTableLen'] = totalLen
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
        parser.add_argument('--template', type=str, default='avr_gcc',
                            help='Using interal template by specifing type.')
        parser.add_argument('--sf2', type=str, default='',
                            help='Sondfont2 file path.')
        parser.add_argument('--sampleName', type=str, default='Celesta C5 Mini',
                            help='Wavetable sample name.')
        parser.add_argument('--outSampleRate', type=int, default=32000,
                            help='Output wavetable sample rate.')
        parser.add_argument('--outSampleWidth', type=int, default=1,
                            help='Wavetable sample wdith.')
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

        if args.sf2 != '':
            (sampleName, attackSamples, loopSamples,
             sampleWidth, sampleRate, sampleChannels) = getFromSf2(args.sf2, args.sampleName)
            attackSamples = samplePipelineProcess(
                attackSamples, sampleChannels, sampleWidth, sampleRate, 1, args.outSampleWidth, args.outSampleRate)
            loopSamples = samplePipelineProcess(
                loopSamples, sampleChannels, sampleWidth, sampleRate, 1, args.outSampleWidth, args.outSampleRate)
            sampleFreq = estimateSampleFreq(
                np.concatenate((attackSamples, loopSamples)), args.outSampleRate)
            genCode(templateFileList, sampleName, sampleFreq, args.outSampleRate,
                    args.outSampleWidth, attackSamples, loopSamples, args.outputDir)
        else:
            pass

    except RuntimeError as identifier:
        print('Meet error during code generation: '+str(identifier))
