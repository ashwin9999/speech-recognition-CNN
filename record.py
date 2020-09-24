import numpy as np
import tensorflow as tf
from test import Test
from speechInput import SingleInputLoader
from speechModel import create_default_model
import preprocess

class Record:

  def run(self):

    """
    This runs the speech recorder.
    importing audio here because we don't want PyAudio to be a requirement 
    if people don't want to do live recording. 
    PyAudio connects to the microphone and is only needed for live recording.
    """
    from audio import AudioRecorder

    loader = SingleInputLoader(128)
    recorder = AudioRecorder()

    with tf.Session() as sess:
      model = create_default_model('record', 128, loader)
      model.restore(sess, 'train/best-weights')
      
      while True:
        print('Listening...')
        audio, width = recorder.record()
        audio = np.array(audio)

        #calculate the power spectrum of the audio and of sampling rate 16000 
        input_ = preprocess.calculatePowerSpectrogram(audio, 16000)

        loader.set_input(input_)
        [decoded] = model.step(sess, loss=False, update=False, decode=True)

        decoded_ids_paths = [Test.extract_decoded_ids(path) for path in decoded]
        
        for decoded_path in decoded_ids_paths:
          decoded_ids = next(decoded_path)
          decoded_str = self.idsToSentence(decoded_ids)
          print('Predicted: {}'.format(decoded_str))

  def idsToSentence(self, identifiers):
    return ''.join(self.idToLetter(identifier) for identifier in identifiers)

  def idToLetter(self, identifier):
    if identifier == 27:
      return ' '
    
    if identifier == 26:
      return '\''
    
    return chr(identifier + ord('a'))
