import wave
import struct
import os
from math import log2, pow
import numpy as np
from sf2utils.sf2parse import Sf2File
from io import StringIO
from string import Template
import struct



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


def getFromSf2(sampleName):
    with open('MusicBox.sf2', 'rb') as sf2_file:
        sf2 = Sf2File(sf2_file)
        for sample in sf2.samples:
            if sample.name==sampleName:
                if sample.sample_rate!=32000:
                    raise RuntimeError('Unsupported sample rate')                    
                if sample.sample_width==1:
                    upackStr='<b'
                elif sample.sample_width==2:
                    upackStr='<h'
                else:
                    raise RuntimeError('Unsupported sample width')
                samples=[ sampleValue[0] for sampleValue in struct.iter_unpack(upackStr, sample.raw_sample_data)]
                attackSamples=samples[0:sample.start_loop]
                loopSamples=samples[sample.start_loop:sample.end_loop]
                return (sampleName,attackSamples,loopSamples,sample.sample_width)              

                


def tmpl_main():
    # sampleName = "Marimba_E5"
    # sampleWidth = 1
    # attackSamples = readWaveSamples("./%s_ATTACK.wav" % sampleName)
    # loopSamples = readWaveSamples("./%s_LOOP.wav" % sampleName)
    # sampleFreq = estimateSampleFreq(attackSamples, 32000)
    # attackLen = len(attackSamples)
    # loopLen = len(loopSamples)

    (sampleName,attackSamples,loopSamples,sampleWidth)=getFromSf2('Xylophone C5')
    sampleFreq = estimateSampleFreq(attackSamples+loopSamples, 32000)
    attackLen = len(attackSamples)
    loopLen = len(loopSamples)
    totalLen = attackLen+loopLen
    sampleWidth=1
    if sampleWidth == 1:
        sampleType = "int8_t"
        attackSamples = [int(sample/256) for sample in attackSamples]
        loopSamples = [int(sample/256) for sample in loopSamples]
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
    paramDict['WaveTableSampleRate'] = 32000
    paramDict['WaveTableLen'] = totalLen
    paramDict['WaveTableAttackLen'] = attackLen
    paramDict['WaveTableLoopLen'] = loopLen
    paramDict['WaveTableSampleType'] = sampleType
    paramDict['WaveTableIncrementType'] = "uint16_t"
    paramDict['WaveTableAttackData'] = attackSamplesDataString
    paramDict['WaveTableLoopData'] = loopSamplesDataString
    paramDict['WaveTableIncrementData'] = incrementDataString
    formatFileByParam('WaveTable.h.avr.template', 'WaveTable.h', paramDict)
    formatFileByParam('WaveTable.c.avr.template', 'WaveTable.c', paramDict)


if __name__ == "__main__":
    tmpl_main()
