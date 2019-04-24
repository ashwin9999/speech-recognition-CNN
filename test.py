
from typing import Dict
import editdistance
import numpy as np
import tensorflow as tf
from testExecutor import TestExecutor
from speechModel import SpeechModel
import itertools

class TestStatistics:

  """
  This class basically is not really needed for the functionalities.
  It is needed to analyze the results of the speech recognition and for the 
  training purposes. 
  Develops statistics like LERs and WERs and LEDs and WEDs based on the paper 
  Wav2Letter.
  """

  def __init__(self):
    self.decodings_counter = 0
    self.sum_letter_edit_distance = 0
    self.sum_letter_error_rate = 0
    self.sum_word_edit_distance = 0
    self.sum_word_error_rate = 0
    self.letter_edit_distance = 0
    self.letter_error_rate = 0
    self.word_edit_distance = 0
    self.word_error_rate = 0

  def track_decoding(self, decoded_str, expected_str):

    """
    Uses editdistance to calculate the distance between the letter differenced between 
    two words. So that we know how many letters were incorrectly predicted.
    Does the same with word using editdistance to find WED.
    Letter Error rate is simply the LED/length of the actual string from transcript.
    Word Error Rate (WER) is the WED/length of actual word.
    """

    self.letter_edit_distance = editdistance.eval(expected_str, decoded_str)
    self.letter_error_rate = self.letter_edit_distance / len(expected_str)
    self.word_edit_distance = editdistance.eval(expected_str.split(), decoded_str.split())
    self.word_error_rate = self.word_edit_distance / len(expected_str.split())
    self.sum_letter_edit_distance += self.letter_edit_distance
    self.sum_letter_error_rate += self.letter_error_rate
    self.sum_word_edit_distance += self.word_edit_distance
    self.sum_word_error_rate += self.word_error_rate
    self.decodings_counter += 1


  """
  These are for the final results after testing on the entire batch.
  """

  @property
  def global_letter_edit_distance(self):
    return self.sum_letter_edit_distance / self.decodings_counter

  @property
  def global_letter_error_rate(self):
    return self.sum_letter_error_rate / self.decodings_counter

  @property
  def global_word_edit_distance(self):
    return self.sum_word_edit_distance / self.decodings_counter

  @property
  def global_word_error_rate(self):
    return self.sum_word_error_rate / self.decodings_counter


class Test(TestExecutor):

  def create_sample_generator(self, limit_count: int):

    """
    Loads the samples from the test directory which has a bunch of audio files
    I am setting the limit count to 0 for now but might have to experiment a bit with it.
    """

    return self.reader.load_samples('test',
                                    loop_infinitely=False,
                                    limit_count=limit_count,
                                    feature_type='power')

  def get_loader_limit_count(self):
    return 0

  def get_max_steps(self):
    if 0:
      return 0
    return None

  def run(self):

    """
    create stats to analyze the results of testing.
    """

    stats = TestStatistics()
    with tf.Session() as sess:
      model = self.create_model(sess) # create a speech model
      coord = self.start_pipeline(sess) 
      try:
        print('Testing on sample audio...')
        if 0:
          step_iter = range(0)
        else:
          step_iter = itertools.count()
        for step in step_iter:
          if coord.should_stop():
            break
          should_save = True and step == 0
          self.run_step(model, sess, stats, should_save)
      except tf.errors.OutOfRangeError:
        print('Testing is done.')
      finally:
        coord.request_stop()
      self.print_global_statistics(stats) #print the overall results of testing (on an entire batch)
      coord.join()

  @staticmethod
  def print_global_statistics(stats):

    """
    Final results in the form of LED and WED (which basically tells how many letters and words were incorrectly predicted)
    """

    print('Final Results')
    print('LED: {} WED: {}'.format(stats.global_letter_edit_distance,stats.global_word_edit_distance))

  def run_step(self, model: SpeechModel, sess: tf.Session, stats: TestStatistics,
               save: bool, verbose=True, feed_dict: Dict=None):

    """
    Probably the hardest function. Has a lot going on but I will try to be as descriptive as possible.
    """

    global_step = model.global_step.eval()
    if save:
      # Validate on the data set and write the summary
      avg_loss, decoded, label, summary = model.step(sess, update=False, decode=True, return_label=True,
                                                     summary=True, feed_dict=feed_dict)
      model.summary_writer.add_summary(summary, global_step)
    else:
      # simply validate, no need to write the summary.
      avg_loss, decoded, label = model.step(sess, update=False, decode=True,
                                            return_label=True, feed_dict=feed_dict)
    decoded_ids_paths = [Test.extract_decoded_ids(path) for path in decoded]
    for label_ids in Test.extract_decoded_ids(label):
      expected_str = self.idsToSentence(label_ids)

      # Print the actual transcript text and the decoded (predicted) text
      # along with it, print the LED and WED so that we know how many letters and 
      # words were incorrectly predicted.
      if verbose:
        print('Actual: {}'.format(expected_str))
      for decoded_path in decoded_ids_paths:
        decoded_ids = next(decoded_path)
        decoded_str = self.idsToSentence(decoded_ids)
        stats.track_decoding(decoded_str, expected_str)
        if verbose:
          print('Predicted: {}'.format(decoded_str))
          print('LED: {} WED: {}'.format(stats.letter_edit_distance,stats.word_edit_distance))

  @staticmethod
  def extract_decoded_ids(sparse_tensor):
    ids = []
    last_batch_id = 0
    for i, index in enumerate(sparse_tensor.indices):
      batch_id, char_id = index
      if batch_id > last_batch_id:
        yield ids
        ids = []
        last_batch_id = batch_id
      ids.append(sparse_tensor.values[i])
    yield ids

  def idsToSentence(self, identifiers):
    return ''.join(self.idToLetter(identifier) for identifier in identifiers)

  def idToLetter(self, identifier):
    if identifier == 27:
      return ' '
    if identifier == 26:
      return '\''
    return chr(identifier + ord('a'))