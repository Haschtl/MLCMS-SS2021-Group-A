# Full example for my blog post at:
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/

input("Tensorflow v1.15 and python3.7 or lower is required! Press Enter to continue...")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tfd = tf.contrib.distributions


def plot_latent_space(ax, epoch, codes, labels, ):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.set_ylabel('Epoch {}'.format(epoch))
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
  for index, sample in enumerate(samples):
    ax[index].imshow(sample, cmap='gray')
    ax[index].axis('off')


def plot_epoch(epoch, codes, labels, samples, num_samples):
  fig, ax = plt.subplots(
      nrows=3, ncols=num_samples, figsize=(10, 20))
  plot_latent_space(ax[0,0], epoch, codes, labels)
  plot_samples(ax[1,:], samples)
  plt.show()
  plt.pause(0.2)

class vae():
  def __init__(self, params):
    make_encoder = tf.make_template('encoder', self.make_encoder)
    make_decoder = tf.make_template('decoder', self.make_decoder)

    self.data = tf.placeholder(tf.float32, [None, *params["input_shape"]])
    self.params = params
    # Define the model.
    self.prior = self.make_prior(params)
    self.posterior = make_encoder(self.data, params)
    self.code = self.posterior.sample()

    # Define the loss.
    likelihood = make_decoder(
        self.code, params).log_prob(self.data)
    divergence = tfd.kl_divergence(self.posterior, self.prior)
    self.elbo = tf.reduce_mean(likelihood - divergence)
    self.optimizer = tf.compat.v1.train.AdamOptimizer(
        params["learning_rate"]).minimize(-self.elbo)

    self.generated_samples = make_decoder(
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


  def make_decoder(self, code, params):
    x = code
    x = tf.layers.dense(x, params["decoder_units"], tf.nn.relu)
    x = tf.layers.dense(x, params["decoder_units"], tf.nn.relu)
    logit = tf.layers.dense(x, np.prod(params["input_shape"]))
    logit = tf.reshape(logit, [-1] + params["input_shape"])
    return tfd.Independent(tfd.Normal(logit, 1.0), 2)
    # return tfd.Independent(tfd.Bernoulli(logit), 2)

  def train(self, dataset):
    plt.ion()
    loss_history = []
    loss_figure = plt.figure()
    loss_axis = loss_figure.add_subplot(1, 1, 1)

    if self.params["continuous_tests"]:
      fig, ax = plt.subplots(
          nrows=self.params["epochs"], ncols=self.params["num_samples"]+1, figsize=(10, 20))
    
    with tf.train.MonitoredSession() as sess:
      for epoch in range(self.params["epochs"]):
        feed = {self.data: dataset.test.images.reshape([-1, *self.params["input_shape"]])}
        test_elbo, test_codes, test_samples = sess.run(
            [self.elbo, self.code, self.generated_samples], feed)
        print('Epoch', epoch, 'elbo', test_elbo)
        
        loss_axis.clear()
        loss_history.append(test_elbo)
        loss_axis.plot(loss_history)
        loss_figure.canvas.draw()
        plt.pause(0.1)
        
        if self.params["continuous_tests"]:
          plot_latent_space(ax[epoch, 0], epoch, test_codes, dataset.test.labels)
          plot_samples(ax[epoch, 1:], test_samples)

        if epoch in self.params["test_after_epochs"]:
          plot_epoch(epoch, test_codes, dataset.test.labels, test_samples, self.params["num_samples"])

        for _ in range(self.params["iterations"]):
          feed = {self.data: dataset.train.next_batch(
              self.params["batch_size"])[0].reshape([-1, *self.params["input_shape"]])}
          sess.run(self.optimizer, feed)
    
    if self.params["continuous_tests"]:
      plt.savefig('task3_training.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.ioff()

def train_mnist(params):
  #  missing: replace Bernoulli
  mnist = input_data.read_data_sets('MNIST_data/')
  model = vae(params)
  model.train(mnist)

  return model, mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

default_params = {
    "input_shape": [28, 28],
    "learning_rate": 0.001,
    "latent_dim": 2,
    "encoder_units": 256,
    "decoder_units": 256,

    "epochs": 100,
    "batch_size": 128,
    "iterations": 60,

    "continuous_tests": False,
    "test_after_epochs": [0,1,5,25,50,100],
    "num_samples": 15,
}

if __name__ == "__main__":
  train_mnist(default_params)
