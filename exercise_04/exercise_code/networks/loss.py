import os
import pickle
import numpy as np


# from exercise_code.networks.linear_model import *


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth, individual_losses=False):
        raise NotImplementedError("Forward pass not implemented in base Loss class")

    def backward(self, y_out, y_truth, upstream_grad=1.):
        raise NotImplementedError("Backward pass not implemented in base Loss class")

    def __call__(self, y_out, y_truth, individual_losses=False):
        loss = self.forward(y_out, y_truth, individual_losses)
        return loss


class L1(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return:
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = np.abs(y_out - y_truth)

        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = np.sign(y_out - y_truth)
        # For y_out - y_truth == 0, np.sign returns 0.
        # Normalize by batch size if this is the final loss gradient for an average loss.
        return gradient / len(y_out)


class MSE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return:
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = (y_out - y_truth) ** 2

        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = 2 * (y_out - y_truth) / len(y_out)  # Normalize by batch size
        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model (the Logits).
        :y_truth: [N, ] array ground truth value of your training set.
        :return:
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss values for each sample of your batch.
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the BCE loss     #
        # for each imstance in the batch.                                      #
        #                                                                      #
        ########################################################################
        epsilon = 1e-12  # To prevent log(0) or log(negative) if y_out is not in (0,1)
        y_out_clipped = np.clip(y_out, epsilon, 1. - epsilon)

        # Standard BCE formula: L = -[y*log(p) + (1-y)*log(1-p)]
        result = -(y_truth * np.log(y_out_clipped) + (1 - y_truth) * np.log(1 - y_out_clipped))
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        if individual_losses:
            return result  # return a list of loss values, without taking the mean.

        return np.mean(result)  # result here is the array of individual losses

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out
                for each sample of your training set.
        """
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t to the input  #
        # to the loss function, y_out.                                         #
        #                                                                      #
        # Hint:                                                                #
        #   Don't forget to divide by N, which is the number of samples in     #
        #   the batch. It is crucial for the magnitude of the gradient.        #
        ########################################################################
        epsilon = 1e-12
        y_out_clipped = np.clip(y_out, epsilon, 1. - epsilon)
        N = y_out.shape[0]

        if N == 0:  # Handle empty batch case
            return np.zeros_like(y_out)

        # Gradient of BCE w.r.t. y_out (p): dL/dp = - (y/p - (1-y)/(1-p))
        # The division by N accounts for the mean if the forward pass computed the mean.
        gradient = - (y_truth / y_out_clipped - (1 - y_truth) / (1 - y_out_clipped)) / N
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return gradient