import numpy as np


class LogisticRegression():
    @staticmethod
    def loss(theta, x, y, lambda_param=None):
        """Loss function for logistic regression with without regularization"""
        # print("theta.shape():", len(theta))
        # print(theta)
        # print("len(x):", len(x))
        # print("len(x)[0]:", len(x[0]))
        # print("x[0]:", x[0])

        # print("Shape of the initial model x: ", x.shape)
        # print("Shape of the initial model theta: ", theta.shape)

        # print("Shape of the initial model y: ", y.shape)

        exponent =  -1 * y * x.dot(theta)

        # print("y[0:10]", y[0:10])
        # print("exponent[0:10]", exponent[0:10])
        # print("exponent max", max(exponent))

        # print("np.sum(np.log(1+np.exp(exponent))) / len(x)", np.sum(np.log(1+np.exp(exponent))) / len(x))
        # print(" len(x)",  len(x))

        return np.sum(np.log(1+np.exp(exponent))) / len(x)

    @staticmethod
    def gradient(theta, x, y, lambda_param=None):
        """
        Gradient function for logistic regression without regularization.
        Based on the above logistic_regression
        """
        exponent = y * (x.dot(theta))
        gradient_loss = - (np.transpose(x) @ (y / (1+np.exp(exponent)))) / (
            x.shape[0])

        # Reshape to handle case where x is csr_matrix
        gradient_loss.reshape(theta.shape)

        return gradient_loss


class LogisticRegressionSinglePoint():
    @staticmethod
    def loss(theta, xi, yi, lambda_param=None):
        exponent = - yi * (xi.dot(theta))
        return np.log(1 + np.exp(exponent))

    @staticmethod
    def gradient(theta, xi, yi, lambda_param=None):

        # Based on page 22 of
        # http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect09.pdf
        exponent = yi * (xi.dot(theta))
        return - (yi*xi) / (1+np.exp(exponent))


class LogisticRegressionRegular():
    @staticmethod
    def loss(theta, x, y, lambda_param):
        regularization = (lambda_param/2) * np.sum(theta*theta)
        return LogisticRegression.loss(theta, x, y) + regularization

    @staticmethod
    def gradient(theta, x, y, lambda_param):
        regularization = lambda_param * theta
        return LogisticRegression.gradient(theta, x, y) + regularization
