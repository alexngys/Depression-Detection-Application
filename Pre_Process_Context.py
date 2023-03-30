import numpy as np
import pandas as pd
import wave
import librosa


train_split_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
test_split_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_label = train_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
test_split_label = test_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()


def extract_features(number, audio_features, label, audio_labels, mode):
    transcript = pd.read_csv('{0}_P/{0}_TRANSCRIPT.csv'.format(number), sep='\t').fillna('') # get transcript
    
    wavefile = wave.open('{0}_P/{0}_AUDIO.wav'.format(number, 'r')) # get wav file
    sr = wavefile.getframerate() # get sr
    nframes = wavefile.getnframes() # get len
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short) # wave file as a np array
    
    start_time = 0
    stop_time = 0

    signal = []
    
    global counter_train # check how many frames for trianing has been obtained

    for t in transcript.itertuples():
        
        if getattr(t,'speaker') == 'Ellie': # pass interviewer
            continue
        elif getattr(t,'speaker') == 'Participant': # if participant is speaking
            if 'scrubbed_entry' in getattr(t,'value'): # reject srcubbed entries maybe add <xxx> rejection?
                continue
            start_time = int(getattr(t,'start_time')*sr) # start pointer for audio extraction
            stop_time = int(getattr(t,'stop_time')*sr) # end pointer for audio extraction
            signal = np.hstack((signal, wave_data[start_time:stop_time].astype(np.float))) # combine extraction with other extractions
    clip = sr*1*6 # set max length to 15 seconds
    if label >= 10 and mode == 'train': # resample if participant is depressed
        times = 3 if counter_train < 48 else 2
        for i in range(times):
            if clip*(i+1) > len(signal): # stop sampling if index exceeds clip
                continue
            #melspec = librosa.feature.melspectrogram(signal[clip*i:clip*(i+1)], n_mels=128, sr=sr) # obtain spectrogram
            #logspec = melspec
            melspec = librosa.feature.melspectrogram(signal[clip*i:clip*(i+1)], n_mels=32, sr=sr, n_fft=2048, hop_length=512,fmin=50,fmax=5000) # obtain spectrogram
            logspec = librosa.amplitude_to_db(np.abs(melspec), ref=np.max) 
            audio_features.append(logspec) # add spectrogram to list
            audio_labels.append(label)  # add label to list
            counter_train+=1
    else:  # no resample
        #melspec = librosa.feature.melspectrogram(signal[:clip], n_mels=128, sr=sr)
        #logspec = melspec
        melspec = librosa.feature.melspectrogram(signal[:clip], n_mels=32, sr=sr, n_fft=2048, hop_length=512,fmin=50,fmax=5000) # obtain spectrogram
        logspec = librosa.amplitude_to_db(np.abs(melspec), ref=np.max)  
        audio_features.append(logspec) 
        audio_labels.append(label)
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
    extract_features(train_split_num[index], audio_features_train, train_split_label[index], audio_labels_train, 'train')


# test set
for index in range(len(test_split_num)):
    extract_features(test_split_num[index], audio_features_test, test_split_label[index], audio_labels_test, 'test')

print(np.shape(audio_labels_train), np.shape(audio_labels_test)) # show overall samples
print(counter_train, counter_test) # show resamples


save_path = 'dataset/'

print("Saving npz file locally...")
np.savez(save_path + 'train_samples_reg.npz', audio_features_train)
np.savez(save_path + 'train_labels_reg.npz', audio_labels_train)
np.savez(save_path + 'test_samples_reg.npz', audio_features_test)
np.savez(save_path + 'test_labels_reg.npz', audio_labels_test)
