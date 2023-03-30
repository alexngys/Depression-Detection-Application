import numpy as np
import pandas as pd
import wave
import librosa
import matplotlib.pyplot as plt
from statistics import mean, median

train_split_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
test_split_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_label = train_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
test_split_label = test_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
train_split_gender = train_split_df[['Gender']]['Gender'].tolist()
test_split_gender = test_split_df[['Gender']]['Gender'].tolist()

def windowing(signal,sr):
    window = 7.5 * sr
    window_diff = window-len(signal)
    window_diff_half = int(window_diff//2)
    signal = np.pad(signal,(window_diff_half,window_diff_half), 'constant', constant_values=(0,0))
    return signal.astype(np.float)

def extract_features(number, audio_features, label, audio_labels, gender): # gender?
    if gender == 1:
        return

    transcript = pd.read_csv('{0}_P/{0}_TRANSCRIPT.csv'.format(number), sep='\t').fillna('') # get transcript
    wavefile = wave.open('{0}_P/{0}_AUDIO.wav'.format(number, 'r')) # get wav file
    sr = wavefile.getframerate() # get sr
    nframes = wavefile.getnframes() # get len
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short) # wave file as a np array
    
    start_time = 0
    stop_time = 0
    count = 0
    for t in transcript.itertuples():
        
        if getattr(t,'speaker') == 'Ellie': # pass interviewer
            continue
        elif getattr(t,'speaker') == 'Participant': # if participant is speaking
            if 'scrubbed_entry' in getattr(t,'value') or '<' in getattr(t,'value'): # reject srcubbed entries maybe add <xxx> rejection?
                continue
            start_time = getattr(t,'start_time') # start pointer for audio extraction
            stop_time = getattr(t,'stop_time') # end pointer for audio extraction
            period = round(stop_time-start_time,3)
            if period < 6 or period >7.5:
                continue
            if count > 15:
                continue
            signal = wave_data[int(start_time*sr):int(stop_time*sr)]
            window_signal = windowing(signal,sr)

            melspec = librosa.feature.melspectrogram(window_signal, n_mels=32, sr=sr, n_fft=2048, hop_length=512,fmin=40,fmax=200) # obtain spectrogram
            logspec = librosa.amplitude_to_db(np.abs(melspec), ref=np.max)
            #mel_spec=librosa.power_to_db(melspec)
            #mfcc = librosa.feature.mfcc(mel_spec)
            #logspec[logspec>-30] = 0
            #logspec[logspec<=-30] = 1
            audio_features.append(logspec) # add spectrogram to list
            audio_labels.append(label)  # add label to list
            count += 1

    print('{}_P feature done'.format(number))



# training set
audio_features_train = []
audio_labels_train = []

# test set
audio_features_test = []
audio_labels_test = []


counter_train = 0
counter_test = 0


# training set
for index in range(len(train_split_num)):
    extract_features(train_split_num[index], audio_features_train, train_split_label[index], audio_labels_train, train_split_gender[index])


# test set
for index in range(len(test_split_num)):
    extract_features(test_split_num[index], audio_features_test, test_split_label[index], audio_labels_test, test_split_gender[index])

print(np.shape(audio_features_train),np.shape(audio_features_test)) # show overall samples
print(np.shape(audio_labels_train),np.shape(audio_labels_test)) # show overall samples
print(counter_train, counter_test) # show resamples


save_path = 'dataset/'

print("Saving npz file locally...")
np.savez(save_path + 'train_samples_reg.npz', audio_features_train)
np.savez(save_path + 'train_labels_reg.npz', audio_labels_train)
np.savez(save_path + 'test_samples_reg.npz', audio_features_test)
np.savez(save_path + 'test_labels_reg.npz', audio_labels_test)
