from sys import byteorder
from array import array
import pyaudio

class AudioRecorder:

  def __init__(self):
    self.rate = 16000
    self.threshold = 0.03 # silence threshold {Need to experiment with it}
    self.chunk_size = 4096
    self.format = pyaudio.paFloat32
    self._pyaudio = pyaudio.PyAudio()

  def isSilent(self, data):
    #Returns true if below the silence threshold
    return max(data) < self.threshold

  def normalize(self, data):
    #Averages the volume {Had to mess with a lot of examples to get this right}
    times = float(0.5) / max(abs(i) for i in data)
    r = array('f')
    for i in data:
      r.append(i * times)

    return r

  def trim(self, data):
    # Trim the blanks at the start and end
    def _trim(data):
      started = False
      r = array('f')
      for i in data:
        if not started and abs(i) > self.threshold:
          started = True
          r.append(i)

        elif started:
          r.append(i)

      return r
    #first trim the left side
    data = _trim(data)
    data.reverse()

    #then trim the right side
    data = _trim(data)
    data.reverse()

    return data

  def addSilence(self, data):
    #adds silence to the start and end of 0.1 seconds
    r = array('f', [0 for i in range(int(0.1 * self.rate))])
    r.extend(data)
    r.extend([0 for i in range(int(0.1 * self.rate))])

    return r

  """
  Records words from the microphone using pyaudio in paFloat32 format and 
  16000Hz sampling rate.
  Returns data as an array of signed floats.
  
  """
  def record(self):
    stream = self._pyaudio.open(format=self.format, channels=1, rate=self.rate, input=True, output=True, frames_per_buffer=self.chunk_size)
    numSilent = 0
    started = False
    r = array('f')
   
    while 1:
      #has to be little endian and signed short
      data = array('f', stream.read(self.chunk_size))
      
      if byteorder == 'big':
        data.byteswap()
      
      r.extend(data)
      silent = self.isSilent(data) #check silent to add blanks
      
      if silent and started:
        numSilent += 1
      
      elif not silent and not started:
        started = True
      
      if started and numSilent > 30: #if there are 30 silences, break. doesnt handle long sentences.
        break
    
    width = self._pyaudio.get_sample_size(self.format)
    stream.stop_stream()
    stream.close()
    r = self.normalize(r) #normalizes the audio
    r = self.trim(r) 
    r = self.addSilence(r)
    
    return r, width

  def terminate(self):
    self._pyaudio.terminate()
