import fnmatch
import logging
import multiprocessing
import os
import random
from multiprocessing.pool import Pool
import librosa
import numpy as np

"""
This function calculates the power spectrum for the supplied raw audio data. 
Audio data needs to be converted into a spectrogram that is basically an array of points in decimals.
It is of the size (x, y) where x = length of the spectrogram and y = no. of mels.
For this particular case, we have chosen a mel size of 128. 
Mel is a perceptual scale of pitches to be equal in distrance from one another.
1 Mel = 2595log10(1 + f/700) Hz

Here, samplerate = sampling rate of the audio data which is 16000Hz in this case.
n_fft = the size of the fft (Fast Fourier Transform)
hop_length = the length of a hop (the number of samples between successive frames)
window = a vector or function used to weight samples within a frame when computing a spectrogram.

"""

def calculatePowerSpectrogram(audio_data, samplerate, n_mels=128, n_fft=512, hop_length=160):

  spect = librosa.feature.melspectrogram(audio_data, sr=samplerate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
  spectLog = librosa.power_to_db(spect, ref=np.max)
  spectNorm = (spectLog - np.mean(spectLog))/(np.std(spectLog))
  print((spectNorm.T).shape)
  return spectNorm.T

def sentenceToIds(sentence):
  return [letterToId(letter) for letter in sentence.lower()]

def letterToId(letter):
  if letter == ' ':
    return 27
  if letter == '\'':
    return 26
  return ord(letter) - ord('a')

"""
This is simply a helper function that recursively searches a directory for a specific 
file pattern for eg (*.flac or *.trans.txt) for audio and transcript files from LibriSpeech.
Returns an iterator for the files found during the recursive traversal.

"""

def recursiveTraverse(directory, file_pattern):
 
  for root, dir_names, file_names in os.walk(directory):
    for filename in fnmatch.filter(file_names, file_pattern):
      yield os.path.join(root, filename)

class DatasetReader:

  """
  This is where reading of preprocessed files are done so that they can be used by the NN.

  """

  def __init__(self, data_directory):

    """
    Creates the reader that will read the samples from the given directory.
    """

    self._data_directory = data_directory
    self._transcript_dict_cache = None

  @property
  def _transcript_dict(self):

    """
    If there is no transcript dict available, it builds one.

    """

    if not self._transcript_dict_cache:
      self._transcript_dict_cache = self._build_transcript()
    return self._transcript_dict_cache

  @staticmethod
  def _get_transcript_entries(transcript_directory):

    """
    It iterates over all the transcript lines and then yields splitted entries. 
    Basically, extracts all the contents for the transcript files in the given directory.
    """
   
    transcript_files = recursiveTraverse(transcript_directory, '*.trans.txt')
    for transcript_file in transcript_files:
      with open(transcript_file, 'r') as f:
        for line in f:
          line = line.rstrip('\n')
          splitted = line.split(' ', 1)
          yield splitted


  def _build_transcript(self):

    """
    Uses the transcript files to map the audio-id to the list of textual ids to
    create a transcript.
    """
  
    transcript_dict = dict()
    for splitted in self._get_transcript_entries(self._data_directory):
      transcript_dict[splitted[0]] = sentenceToIds(splitted[1])
    return transcript_dict

  @classmethod
  def _extract_audio_id(cls, audio_file):

    """
    Given the audio file, it provides the audio id to link.
    """

    file_name = os.path.basename(audio_file)
    audio_id = os.path.splitext(file_name)[0]
    return audio_id

  @classmethod
  def _transform_sample(cls, audio_file, preprocess_fnc):

    """
    Librosa is a tool that is used to transform audio files into features.
    For this case, I am using a feature called Power Spectrum as suggested in the 
    Wav2Letter paper referenced. I can use other features like MFCCs with it but 
    Power spectrum was easier to generate. 
    I am transforming the audio file into audio fragments using power spectrum and then returning the 
    fragment along with the audio id.
    """

    audio_data, samplerate = librosa.load(audio_file)
    audio_fragments = preprocess_fnc(audio_data, samplerate)
    audio_id = cls._extract_audio_id(audio_file)
    return audio_id, audio_fragments

  @classmethod
  def _transform_and_store_sample(cls, audio_file, preprocess_fnc, transcript, out_directory):

    """
    I am saving the transcript [audio id and audio fragment] into an .npz file.
    .npz file format is a zipped archive of files. It is not compressed and each file in this format
    contains one variable in .npy format [numpy lib format] [Dictionary-like object]
    """

    audio_id, audio_fragments = cls._transform_sample(audio_file, preprocess_fnc)
    np.savez(out_directory + '/' + audio_id, audio_fragments=audio_fragments, transcript=transcript)

  def generate_samples(self, directory, preprocess_fnc):

    """
    Basically does the same as before. Generates audio fragments using power spectrum for all the flac audio files
    in the subdirectories. 
    Returns a generator with audio id which is a string, fragments which is a ndarray and 
    transcript with is a list of int 

    """

    audio_files = list(recursiveTraverse(self._data_directory + '/' + directory, '*.flac'))
    transcript_dict = self._transcript_dict
    for audio_file in audio_files:
      audio_id, audio_fragments = self._transform_sample(audio_file, preprocess_fnc)
      if (audio_id in transcript_dict):
        yield audio_id, audio_fragments, transcript_dict[audio_id]

  def _get_directory(self, feature_type, sub_directory):

    """
    This simply returns the directory where the preprocessed data will be stored.
    """
    preprocess_directory = 'preprocessed'
    directory = self._data_directory + '/' + preprocess_directory + '/' + sub_directory
    return directory

  @classmethod
  def _preprocessing_error_callback(cls, error: Exception):
    raise RuntimeError('Error during preprocessing') from error

  def store_samples(self, directory, preprocess_fnc):

    """
    Reads audio files from the directory supplied and stores the preprocessed files
    into the preprocessed/ directory. 
    Pretty straightforward
    """
  
    out_directory = self._get_directory(preprocess_fnc, directory)
    if not os.path.exists(out_directory):
      os.makedirs(out_directory)
    audio_files = list(recursiveTraverse(self._data_directory + '/' + directory, '*.flac'))
    with Pool(processes=multiprocessing.cpu_count()) as pool:
      transcript_dict = self._transcript_dict
      for audio_file in audio_files:
        audio_id = self._extract_audio_id(audio_file)
        if (audio_id in transcript_dict):
          transcript_entry = transcript_dict[audio_id]
        transform_args = (audio_file, preprocess_fnc, transcript_entry, out_directory)
        pool.apply_async(DatasetReader._transform_and_store_sample, transform_args,
                         error_callback=self._preprocessing_error_callback)
      pool.close()
      pool.join()

  def load_samples(self, directory, max_size=False, loop_infinitely=False, limit_count=0, feature_type='mfcc'):


    """
    Now, once the preprocessing is done. The preprocessed samples need to be loaded
    for training, testing and recording. 
    directory = dir to use
    max_size = basically asking if there should be a maximum audio length and if there is one, everything else will be removed
    default is set to false for max_size because we dont want a max value
    loop_infinitely = asking if after one pass, do we want to shuffle and pass again. 
    limit_count: max number of samples to use (default is 0 because in this case, 0 means unlimited.

    """

    load_directory = self._get_directory(feature_type, directory)
    if not os.path.exists(load_directory):
      raise ValueError('Directory {} does not exist'.format(load_directory))
    files = list(recursiveTraverse(load_directory, '*.npz'))
    random.shuffle(files)
    if limit_count:
      files = files[:limit_count]
    while True:
      for file in files:
        with np.load(file) as data:
          audio_length = data['audio_fragments'].shape[0]
          if not max_size or audio_length <= max_size:
            yield data['audio_fragments'], data['transcript']
          else:
            logging.warning('Audio snippet too long: {}'.format(audio_length))
      if not loop_infinitely:
        break
      random.shuffle(files)

    

class Preprocess:

  def run(self):

    """
    Basically, runs the data reader to generate the power spectrum and preprocess the data 
    using the mechanisms given above.
    """

    reader = DatasetReader('data')
    preprocess_fnc = calculatePowerSpectrogram
    reader.store_samples('train', preprocess_fnc)
    reader.store_samples('test', preprocess_fnc)

   
