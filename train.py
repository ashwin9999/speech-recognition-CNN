import os
import time
import numpy as np
import tensorflow as tf
from testExecutor import TestExecutor
from speechModel import create_default_model

class Train(TestExecutor):

  # Loads the samples from the preprocessed train directory that has the npz files to train with

  def create_sample_generator(self, limit_count: int):
  	self.limit = limit_count
  	return self.reader.load_samples('train', loop_infinitely=True,limit_count=limit_count,feature_type='power')

  def get_loader_limit_count(self) -> int:
    return self.limit

  def create_model(self, sess):

    # creates a model for training, using the best-weights saved and a learning rate of 1e-4
    model = create_default_model('train', self.input_size, self.speech_input)
    model.restore_or_create(sess,'train/best-weights',1e-4)
    return model

  def run(self):

    with tf.Session() as sess:

      model = self.create_model(sess)
      coord = self.start_pipeline(sess, n_threads=2)
      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []
      try:
        print('Begin training')
        while not coord.should_stop():

          current_step += 1
          is_checkpoint_step = current_step % 1000 == 0

          start_time = time.time()
          step_result = model.step(sess, summary=is_checkpoint_step)
          avg_loss = step_result[0]
          step_time += (time.time() - start_time) / 1000
          loss += avg_loss / 1000

          # save the checkpoint and print the stats

          if is_checkpoint_step:
            global_step = model.global_step.eval()

            # prints the stats for the previous step
            perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
            print("global step {:d} learning rate {:.4f} step-time {:.2f} average loss {:.2f} perplexity {:.2f}"
                  .format(global_step, model.learning_rate.eval(), step_time, avg_loss, perplexity))
            
            # store the summary
            summary = step_result[2]
            model.summary_writer.add_summary(summary, global_step)
            previous_losses.append(loss)

            #save the checkpoint inside the weights directory for faster access later
            checkpoint_path = os.path.join('train/best-weights', "speech.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print('Weights saved')
            step_time, loss = 0.0, 0.0

      except tf.errors.OutOfRangeError:
        print('Done training.')
      finally:
        coord.request_stop()

      coord.join()

