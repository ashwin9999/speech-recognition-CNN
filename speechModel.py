import abc
import math
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from speechInput import BaseInputLoader

class SpeechModel:

  def __init__(self, input_loader: BaseInputLoader, input_size: int, num_classes: int):

    """
    Probably the most important class that does most of the tf and nn stuff.
    Creates a new speech model which is required for training, testing and recording.
    input_loader : provides the input tensors
    input_size : number of values per time step
    num_classes : the number of output classes (29) [a-z = 26 + 1 (space) + 1 (apostrophe) + 1 (blank label) Refer to CTC]
    """
   
    self.input_loader = input_loader
    self.input_size = input_size
    self.convolution_count = 0
    self.global_step = tf.Variable(0, trainable=False)

    # inputs is of dimension [batch size, max time, input size] 
    self.inputs, self.sequence_lengths, self.labels = input_loader.get_inputs()
    self.logits = self._create_network(num_classes)


    # For summaries [image and histogram for logits [batch size, height = num of classes, width=max time /2, channels = 1]]
    tf.summary.image('logits', tf.expand_dims(tf.transpose(self.logits, (1, 2, 0)), 3))
    tf.summary.histogram('logits', self.logits)

  def add_training_ops(self, learning_rate: bool = 1e-3, learning_rate_decay_factor: float = 0,
                       max_gradient_norm: float = 5.0, momentum: float = 0.9):

    """
    Hyperparamaters for training
    learning rate = 1e-3 (need to experiment with => this gave the best results so far)
    learning rate decay factor = the factor to multiply the learning rate with when you need to decrease it
    max gradient norm = maximum gradient norm to apply, otherwise clipping is needed
    momentum = momentum for the optimizer (default set to 0.9)
    """
   
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

    """
    Variable summaries
    """
    tf.summary.scalar('learning_rate', self.learning_rate)

    # Loss and optimizer
    if self.labels is not None:
      with tf.name_scope('training'):
        self.cost = tf.nn.ctc_loss(self.labels, self.logits, self.sequence_lengths // 2) #using ctc loss (refer to ctc and wav2letter paper)
        self.avg_loss = tf.reduce_mean(self.cost, name='average_loss')
        tf.summary.scalar('loss', self.avg_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-3) #using AdamOptimizer {gave the best results}
        gvs = optimizer.compute_gradients(self.avg_loss)
        gradients, trainables = zip(*gvs)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm, name='clip_gradients')
        self.update = optimizer.apply_gradients(zip(clipped_gradients, trainables),
                                                global_step=self.global_step, name='apply_gradients')

  def add_decoding_ops(self, language_model: str = None, lm_weight: float = 0.8, word_count_weight: float = 0.0,
                       valid_word_count_weight: float = 2.3):
    
    """
    For decoding the sound wave and predicting.
    lanuage model : we are not using any langugage model but we can use models like kenlm if needed and use 
    a beam search decoder instead of a greedy decoder for ctc
    word count weight : Weight added for each word
    valid word count weight : weight added for each word in vocabulary
    lm weight : weight multiplied with the language model scoring
    """

    with tf.name_scope('decoding'):

      # These are only needed if we use a language model
      self.lm_weight = tf.placeholder_with_default(lm_weight, shape=(), name='language_model_weight')
      self.word_count_weight = tf.placeholder_with_default(word_count_weight, shape=(), name='word_count_weight')
      self.valid_word_count_weight = tf.placeholder_with_default(valid_word_count_weight, shape=(),
                                                     name='valid_word_count_weight')
      # use a ctc greedy decoder to determine the decoded string
      """
      If merge repeated = True, this is how greedy decoder works:
      Sequence: A B B * B * B ( where * is the blank label ) becomes A B B B 
      If merge repeated = False, it becomes A B B B B (so merging is not done)
      For my case, I want it to merge to decode the string.
      """
      self.decoded, self.log_probabilities = tf.nn.ctc_greedy_decoder(self.logits,
                                                                        self.sequence_lengths // 2,
                                                                        merge_repeated=True)

  def finalize(self, log_dir: str, run_name: str, run_type: str):

    # Initialize the variables
    self.init = tf.global_variables_initializer()

    # Create a tf saver
    self.saver = tf.train.Saver(tf.global_variables()) # Saves and restores variables to and from checkpoints

    # Create summary writers for logging
    self.merged_summaries = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter('{}/{}_{}'.format(log_dir, run_name, run_type))

  def _convolution(self, value, filter_width, stride, input_channels, out_channels, apply_non_linearity=True):

    """
    Builds the convolutional layers (wav2letter paper)

    value: input tensor to apply the convolution on (A 3D tensor => float16, float32, float64)
    filter width : width of the filter (kernel)
    stride : striding of the filter (kernel) (An integer => no. of entries by which the filter is moved right at each step)
    input channels : number of input channels
    output channels : number of output channels
    apply non linearity : whether to apply a non linearity
    
    Returns the convolutional output (adds biases)

    """

    layer_id = self.convolution_count
    self.convolution_count += 1
    with tf.variable_scope('convolution_layer_{}'.format(layer_id)) as layer:

      # Creates variables filter and bias
      filters = tf.get_variable('filters', shape=[filter_width, input_channels, out_channels],
                                dtype=tf.float32, initializer=xavier_initializer())
      bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='bias')

      # Applies convolution (using a 1d standard convNN) padding = 'SAME', 
      # Builds a 1-D convolution given 3-D input and filter tensors.
      convolution_out = tf.nn.conv1d(value, filters, stride, 'SAME', use_cudnn_on_gpu=True, name='convolution')

      # Creates the summaries for logging
      with tf.name_scope('summaries'):

        # add depth of 1 makes the space [filter width, input channels, 1, output channels]
        kernel_with_depth = tf.expand_dims(filters, 2)

        # tf.image_summary [batch_Size = output chanels, height = filter width, width = input channels, channels = 1]

        kernel_transposed = tf.transpose(kernel_with_depth, [3, 0, 1, 2])


        # This will display random 3 filters from all the output channels

        tf.summary.image(layer.name + 'filters', kernel_transposed, max_outputs=3)
        tf.summary.histogram(layer.name + 'filters', filters)
        tf.summary.image(layer.name + 'bias', tf.reshape(bias, [1, 1, out_channels, 1]))
        tf.summary.histogram(layer.name + 'bias', bias)

      # adds the biases

      convolution_out = tf.nn.bias_add(convolution_out, bias)
      if apply_non_linearity:

        # uses relu activation function (best results) to add non-linearity
        activations = tf.nn.relu(convolution_out, name='activation')
        tf.summary.histogram(layer.name + 'activation', activations)
        return activations, out_channels
      else:
        return convolution_out, out_channels

  def init_session(self, sess, init_variables=True):

    """
    Initializes a new session for the model.
    """

    if init_variables:
      sess.run(self.init)
    self.summary_writer.add_graph(sess.graph)

  def step(self, sess, loss=True, update=True, decode=False, return_label=False, summary=False, feed_dict=None):

    """
    sess = tensorflow session
    update = if the network is to be trained
    loss = if outputing the average loss is required
    decode = if decoding is to be done
    return label = is the label needs to be return
    summary = if the summary needs to be generated
    feed dict = additional tensors that can be fed

    """

    output_feed = []
    if loss:
      output_feed.append(self.avg_loss)
    if decode:
      output_feed.append(self.decoded)
    if return_label:
      output_feed.append(self.labels)
    if update:
      output_feed.append(self.update)
    if summary:
      output_feed.append(self.merged_summaries)
    input_feed_dict = self.input_loader.get_feed_dict() or {}
    if feed_dict is not None:
      input_feed_dict.update(feed_dict)
    return sess.run(output_feed, feed_dict=input_feed_dict)

  @abc.abstractclassmethod
  def _create_network(self, num_classes):
    raise NotImplementedError()

  def restore(self, session, checkpoint_directory: str, reset_learning_rate: float = None):

    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(session, ckpt.model_checkpoint_path)
      self.init_session(session, init_variables=False)
      if reset_learning_rate:
        session.run(self.learning_rate.assign(reset_learning_rate))
    else:
      raise FileNotFoundError('No checkpoint for evaluation found')

  def restore_or_create(self, session, checkpoint_directory: str, reset_learning_rate: float = None):
    try:
      self.restore(session, checkpoint_directory, reset_learning_rate)
    except FileNotFoundError:
      self.init_session(session, init_variables=True)


class Wav2LetterModel(SpeechModel):

  def __init__(self, input_loader: BaseInputLoader, input_size: int, num_classes: int):
    super().__init__(input_loader, input_size, num_classes)

  def _create_network(self, num_classes):

    """
    So for our case, I am using the Wav2Letter speech model from the paper referenced.
    for this particular model, I have to first create a network.

    """
 
    # The first layer scales up from 128 (input size) to 250 channels.
    # One striding layer of output size [batch size, maxtime/2, 250]
    outputs, channels = self._convolution(self.inputs, 48, 2, self.input_size, 250)
    
    # 7 layers without striding of output size [batch size, max time/2, 250]
    for layer_idx in range(7):
      outputs, channels = self._convolution(outputs, 7, 1, channels, channels)
    
    # 1 layer with high kenerl width and output size [batch size, max time/2, 2000]
    outputs, channels = self._convolution(outputs, 32, 1, channels, channels * 8)

    # 1 fully connected layer of output size [batch size, max time/2, 2000]
    outputs, channels = self._convolution(outputs, 1, 1, channels, channels)

    # 1 fully connected layer of output size [batch size, max time/2, number of classes]
    # Last layer must have non linearity
    outputs, channels = self._convolution(outputs, 1, 1, channels, num_classes, False)

    # transpose logits to size [max time/2, batch size, number of classes]
    return tf.transpose(outputs, (1, 0, 2))


def create_default_model(command, input_size: int, speech_input: BaseInputLoader) -> SpeechModel:
  model = Wav2LetterModel(input_loader=speech_input,
                          input_size=input_size,
                          num_classes=29)
  if command == 'train':
    model.add_training_ops(learning_rate=flags.learning_rate,
                           learning_rate_decay_factor=flags.learning_rate_decay_factor,
                           max_gradient_norm=flags.max_gradient_norm,
                           momentum=flags.momentum)
    model.add_decoding_ops()
  else:
    model.add_training_ops()
    model.add_decoding_ops(language_model=None,
                           lm_weight=0.8,
                           word_count_weight=0.0,
                           valid_word_count_weight=2.3)

  model.finalize(log_dir='log',
                 run_name='best-weights',
                 run_type='train')

  return model
