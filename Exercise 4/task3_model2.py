# Full example for my blog post at:
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/

input("Tensorflow v1.15 and python3.7 or lower is required! Press Enter to continue...")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tfd = tf.contrib.distributions


def concat_images(imga, imgb):
  """
  Combines two color image ndarrays side-by-side.
  """
  ha, wa = imga.shape[:2]
  hb, wb = imgb.shape[:2]
  max_height = np.max([ha, hb])
  total_width = wa+wb
  new_img = np.zeros(shape=(max_height, total_width))
  new_img[:ha, :wa] = imga
  new_img[:hb, wa:wa+wb] = imgb
  return new_img

def plot_loss_history(fig, ax, data):
  ax.clear()
  ax.plot(data)
  ax.set_xlabel("Epoch")
  ax.set_ylabel("$L_{ELBO}$")
  fig.canvas.draw()
  # plt.show()
  plt.pause(0.1)

def plot_latent_space_2d(ax, epoch, codes, labels, ):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.set_title('Epoch {}'.format(epoch))
  ax.set_xlabel('Latent dim 1')
  ax.set_ylabel('Latent dim 2')
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_latent_space_3d(ax, epoch, codes, labels, ):
  ax.scatter(codes[:, 0], codes[:, 1], codes[:, 2],s=2, c=labels, alpha=0.1)
  ax.set_aspect('auto')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.set_zlim(codes.min() - .1, codes.max() + .1)
  ax.set_title('Epoch {}'.format(epoch))
  ax.set_xlabel('Latent dim 1')
  ax.set_ylabel('Latent dim 2')
  ax.set_zlabel('Latent dim 3')
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_reconstructed_samples(ax, original_samples, reconstructed_samples):
  # print(original_samples)
  # print(reconstructed_samples)
  for idx, sample in enumerate(reconstructed_samples):
    # ax[idx].imshow(sample, cmap='gray')
    try:
      ax[idx].imshow(concat_images(original_samples[idx], sample), cmap='gray')
      ax[idx].axis('off')
    except Exception:
      pass


def plot_generated_samples(ax, generated_samples):
  # print(original_samples)
  # print(reconstructed_samples)
  for idx, sample in enumerate(generated_samples):
    # ax[idx].imshow(sample, cmap='gray')
    try:
      ax[idx].imshow(sample, cmap='gray')
      ax[idx].axis('off')
    except Exception:
      pass
  

def plot_epoch(epoch, params, codes, labels, original_samples, reconstructed_samples, generated_samples, num_samples):
  fig = plt.figure(figsize=(6,10))
  if (params["latent_dim"] == 2 or params["latent_dim"] == 3):
    columns=5
    rows = 3+3+1+1
    if params["latent_dim"] == 2:
      ax1 = plt.subplot2grid((columns+rows+1, columns), (1, 0),
                            colspan=columns, rowspan=columns)
    else:
      ax1 = plt.subplot2grid((columns+rows+1, columns), (1, 0),
                             colspan=columns, rowspan=columns, projection='3d')
    yoffset = columns
  else:
    columns = 5
    rows = 3+3+1+1
    yoffset = 0
  ax2 = [
      *[plt.subplot2grid((yoffset+rows+1, columns), (yoffset+2, i)) for i in range(columns)],
      *[plt.subplot2grid((yoffset+rows+1, columns), (yoffset+3, i)) for i in range(columns)],
      *[plt.subplot2grid((yoffset+rows+1, columns), (yoffset+4, i)) for i in range(columns)],
  ]
  ax3 = [
      *[plt.subplot2grid((yoffset+rows+1, columns), (yoffset+6, i)) for i in range(columns)],
      *[plt.subplot2grid((yoffset+rows+1, columns), (yoffset+7, i)) for i in range(columns)],
      *[plt.subplot2grid((yoffset+rows+1, columns), (yoffset+8, i)) for i in range(columns)],
  ]
  if (params["latent_dim"] == 2):
    plot_latent_space_2d(ax1, epoch, codes, labels)
  if (params["latent_dim"] == 3):
    plot_latent_space_3d(ax1, epoch, codes, labels)
  ax2[2].set_title("Reconstructed samples")
  plot_reconstructed_samples(ax2, original_samples, reconstructed_samples)
  ax3[2].set_title("Generated samples", pad=20)
  plot_generated_samples(ax3, generated_samples)
  plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
  # fig.tight_layout()
  plt.show()
  plt.pause(0.2)
  fig.savefig('task3_epoch{}_latentdim{}.png'.format(epoch, params["latent_dim"]), dpi=100)

class vae():
  def __init__(self, params):
    self._decoder_standard_deviation = tf.Variable(1., trainable=True)
    # self._decoder_standard_deviation = tf.make_template('dsd',decoder_standard_deviation)
    make_encoder = tf.make_template('encoder', self.make_encoder)
    make_decoder = tf.make_template('decoder', self.make_decoder)

    self.data = tf.placeholder(tf.float32, [None, *params["input_shape"]])
    self.params = params
    # Define the model.
    self.prior = self.make_prior(params)
    self.posterior = make_encoder(self.data, params)
    self.latent_space = self.posterior.sample()

    # Define the loss.
    samples2 = make_decoder(
        self.latent_space, params)
    self.likelihood = samples2.log_prob(self.data)
    divergence = tfd.kl_divergence(self.posterior, self.prior)
    self.elbo = tf.reduce_mean(self.likelihood - divergence)
    self.optimizer = tf.compat.v1.train.AdamOptimizer(
        params["learning_rate"]).minimize(-self.elbo)

    self.samples2 = samples2.mean()
    self.samples = make_decoder(
        self.prior.sample(params["num_samples"]), params).mean()


  def make_encoder(self, data, params):
    x = tf.layers.flatten(data)
    x = tf.layers.dense(x, params["encoder_units"], tf.nn.relu)
    x = tf.layers.dense(x, params["encoder_units"], tf.nn.relu)
    loc = tf.layers.dense(x, params["latent_dim"])
    scale = tf.layers.dense(x, params["latent_dim"], tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale)


  def make_prior(self, params):
    loc = tf.zeros(params["latent_dim"])
    scale = tf.ones(params["latent_dim"])
    return tfd.MultivariateNormalDiag(loc, scale)


  def make_decoder(self, latent_space, params):
    x = latent_space
    x = tf.layers.dense(x, params["decoder_units"], tf.nn.relu)
    x = tf.layers.dense(x, params["decoder_units"], tf.nn.relu)
    logit = tf.layers.dense(x, np.prod(params["input_shape"]))
    logit = tf.reshape(logit, [-1] + params["input_shape"])
    return tfd.Independent(tfd.Normal(logit, self._decoder_standard_deviation), 2)
    # return tfd.Independent(tfd.Bernoulli(logit), 2)

  def train(self, dataset):
    plt.ion()
    loss_history = []
    loss_figure = plt.figure()
    loss_axis = loss_figure.add_subplot(1, 1, 1)

    with tf.train.MonitoredSession() as sess:
      for epoch in range(self.params["epochs"]+1):
        if epoch != 0:
          for _ in range(self.params["iterations"]):
            feed = {self.data: dataset.train.next_batch(
                self.params["batch_size"])[0].reshape([-1, *self.params["input_shape"]])}
            sess.run(self.optimizer, feed)
        
        feed = {self.data: dataset.test.images.reshape([-1, *self.params["input_shape"]])}
        test_elbo, test_codes, test_samples = sess.run(
            [self.elbo, self.latent_space, self.samples], feed)
        print('Epoch {}; Elbo: {}'.format(epoch, test_elbo))

        feed = {self.data: dataset.test.images.reshape(
            [-1, *self.params["input_shape"]])}
        _, _, reconstructed_samples = sess.run(
            [self.elbo, self.latent_space, self.samples2], feed)

        loss_history.append(test_elbo)
        plot_loss_history(loss_figure, loss_axis, loss_history)
        
        if epoch in self.params["test_after_epochs"]:
          plot_epoch(epoch, self.params, test_codes, dataset.test.labels, feed[self.data], reconstructed_samples, 
                     test_samples, self.params["num_samples"])
    
    loss_figure.savefig('task3_L_elbo_latentdim{}.png'.format(self.params["latent_dim"]), dpi=100)
    plt.ioff()

def train_mnist(params):
  mnist = input_data.read_data_sets('MNIST_data/')
  model = vae(params)
  model.train(mnist)

  return model, mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

default_params = {
    "input_shape": [28, 28],
    "learning_rate": 0.001,
    "latent_dim": 32,
    "encoder_units": 256,
    "decoder_units": 256,

    "epochs": 500,
    "batch_size": 128,
    "iterations": 60,

    "test_after_epochs": [1,5,25,50,100,150,200,250,300,350,400,450,500],
    "num_samples": 15,
}

if __name__ == "__main__":
  train_mnist(default_params)
