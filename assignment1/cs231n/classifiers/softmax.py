import numpy as np
from random import shuffle
from past.builtins import xrange
from tqdm import tqdm

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  class_num = W.shape[1]

  for i in tqdm(xrange(train_num)):
      scores = X[i].dot(W)
      shift_scores = scores - np.max(scores)
      shift_scores_sum = np.sum(np.exp(shift_scores))
      loss_i = - shift_scores[y[i]] + np.log(shift_scores_sum)
      loss += loss_i
      for j in xrange(class_num):
          softmax_output = np.exp(shift_scores[j]) / shift_scores_sum
          if j == y[i]:
              dW[:, j] += (-1 + softmax_output) * X[i]
          else:
              dW[:, j] += softmax_output * X[i]

  loss = loss / train_num + reg * np.sum(W*W)
  dW = dW / train_num + 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  train_num = X.shape[0]
  feature_num = X.shape[1]
  class_num = W.shape[1]


  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis=1, keepdims=True)

  shift_scores_sum = np.sum(np.exp(shift_scores), axis=1, keepdims=True)
  loss_x = (
          - shift_scores[range(train_num), list(y)] +
          np.log(shift_scores_sum)
          )

  loss = np.sum(loss_x) / train_num + reg * np.sum(W*W)

  softmax_output = (np.exp(shift_scores) / shift_scores_sum)
  softmax_output[range(train_num), list(y)] -= 1
  dW = X.T.dot(softmax_output)

  loss = loss / train_num + reg * np.sum(W*W)
  dW = dW / train_num + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

