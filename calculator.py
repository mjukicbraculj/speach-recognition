from array import array
from struct import pack
from sys import byteorder
import pyaudio
import wave
import random
import string
import os
import librosa
import multiprocessing
import dtw
import numpy
import webrtcvad
import tkinter


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """for generating random name for recorded audio file"""
    return ''.join(random.choice(chars) for _ in range(size))

#number is for setting mode of aggresivness of filtering speech / non-speech
vad = webrtcvad.Vad(2)

THRESHOLD = 900
CHUNK_SIZE = 1024
SILENT_CHUNKS = 2 * 44000 / 1024  # about 2 sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20) #
RATE = 44100        #sampling rate  (s^-1)
CHANNELS = 1
TRIM_APPEND = RATE / 4

characters = {}
characters["nula"] = 0
characters["jedan"] = 1
characters["dva"] = 2
characters["tri"] = 3
characters["cetiri"] = 4
characters["pet"] = 5
characters["sest"] = 6
characters["sedam"] = 7
characters["osam"] = 8
characters["devet"] = 9
characters["plus"] = "+"
characters["minus"] = "-"
characters["puta"] = "*"
characters["dijeljeno"] = "/"
characters["jednako"] = "="

screen = tkinter.Tk()
text = tkinter.Text(screen)
text.pack()


def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD

# def is_speach(data_chunk):
#     return vad.is_speech(data_chunk, RATE)


def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r


def trim(data_all):
    """Trim silence from begining and end of data"""
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return data_all[int(_from):int(_to + 1)]


def record(q):
    """Record a word or words from the microphone"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    silent_chunks = 0
    audio_started = False
    there_was_noise = False
    data_all = array('h')

    while True:
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)

        silent = is_silent(data_chunk)
        if not silent:
            there_was_noise = True

        if audio_started:
            if silent:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    if there_was_noise:
                        there_was_noise = False
                        file = 'words/'+id_generator(8)+'.wav'
                        data = trim(data_all)
                        record_to_file(file, p.get_sample_size(FORMAT), normalize(trim(data)))
                        q.put(file)
                        print("snimljen file:", file)
                    data_all = array('h')
                    silent_chunks = 0
            else:
                silent_chunks = 0
        elif not silent:
            audio_started = True

    stream.stop_stream()
    stream.close()
    p.terminate()


def record_to_file(path, sample_width, data):
    "Records from the microphone and outputs the resulting data to 'path'"
    data = pack('<' + ('h' * len(data)), *data)

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()


def recognize(q):
    train_mfccs = {}
    dtw_distances = open("words/dtw_distance.txt", "w")
    train_data_path = "DATASET/"
    train_file_list = os.listdir(train_data_path)

    if os.path.isfile("words/train_mfcss.txt"):
        train_mfccs_file = open("words/train_mfcss.txt", "r")
        one_word_mfccs = train_mfccs_file.read().split('**')
        for index, name in enumerate(train_file_list):
            one_chunk_mfccs = one_word_mfccs[index].split("//")
            train_mfccs[name] = []
            for temp_list in one_chunk_mfccs:
                train_mfccs[name].append(list(map(float, temp_list.split(","))))
    else:
        train_mfccs_file = open("words/train_mfcss.txt", "w")
        for i in range(len(train_file_list)):
            data, sampling_rate = librosa.load(train_data_path + "/" + train_file_list[i])
            mfcc = librosa.feature.mfcc(data, sampling_rate, n_mfcc=13)
            train_mfccs[train_file_list[i]] = mfcc.T
            for j, mfcc_list in enumerate(mfcc.T):
                for k, item in enumerate(mfcc_list):
                    if k != 0:
                        train_mfccs_file.write(","+str(item))
                    else:
                        train_mfccs_file.write(str(item))
                if j != len(mfcc.T)-1:
                    train_mfccs_file.write("//")
            train_mfccs_file.write("**")
        print("IZRACUNAO MFCC!")
        train_mfccs_file.close()

    expression = ""
    while True:
        file = q.get()
        test_data, test_sampling_rate = librosa.load(file)
        test_mfcc = librosa.feature.mfcc(test_data, test_sampling_rate, n_mfcc=13).T
        first = True
        for i in range(len(train_mfccs.keys())):
            dtw_distance, _, _, _ = dtw.dtw(test_mfcc, train_mfccs[list(train_mfccs.keys())[i]], dist=lambda x, y: numpy.linalg.norm(x - y, ord=1))
            dtw_distances.write(str(dtw_distance)+",")
            if first or min_distance > dtw_distance:
                first = False
                min_distance = dtw_distance
                min_label = list(train_mfccs.keys())[i]
        dtw_distances.write("\n-------------------------\n")
        print("izgovorena riječ je: ", min_label)
        char = min_label.split("_")[0]
        expression += str(characters[char])
        text.insert(tkinter.END, characters[char])
        screen.update()
        if characters[char] == '=':
            result = evaluate(expression)
            text.insert(tkinter.END, result)
            screen.update()


def evaluate(expression):
    return "Ne da mi se racunati, sutra cu :)\n"


def main():
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(name="p2", target=record, args=(q,))
    p2 = multiprocessing.Process(name="p1", target=recognize, args=(q,))
    p1.start()
    p2.start()

    screen.mainloop()

if __name__ == '__main__':
    main()