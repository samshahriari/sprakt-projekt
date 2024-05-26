from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import codecs


"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta
            print('Model parameters:');

            print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))


        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        x = self.x[datapoint]
        prob = self.sigmoid(np.dot(self.theta,x))
        if label==1:
            return prob
        else:
            return 1-prob
        

    def minibatch_fit_with_early_stopping(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.LEARNING_RATE = 0.01
        iteration = 0
        print("Starting Mini-Batch Fit...")
        num_batches = self.DATAPOINTS // self.MINIBATCH_SIZE
        weight = {0 : 0.4, 1: 0.6} # Apply weights as needed
        max_iter = 50
        old_grad = 1
        increase_counter=0
        while iteration < max_iter:
            if iteration%1==0:
                print(np.sum(np.square(self.gradient)))
                print(f"iteration: {iteration}")
            iteration += 1
            #print(f"num batches: {num_batches}")
            for batch_i in range(num_batches):
                start_i = batch_i * self.MINIBATCH_SIZE
                end_i = start_i + self.MINIBATCH_SIZE
                x_batch = self.x[start_i:end_i]
                y_batch = self.y[start_i:end_i]
                self.gradient = np.zeros(self.FEATURES)
                for i in range(self.MINIBATCH_SIZE):
                    xi = x_batch[i]
                    yi = y_batch[i]
                    h = self.sigmoid(np.dot(self.theta, xi)) - yi
                    error = h * weight[yi]
                    self.gradient += xi * error
                self.gradient /= self.MINIBATCH_SIZE
                
                self.theta -= self.LEARNING_RATE * self.gradient
            else:
                increase_counter = 0
            old_grad = np.sum(np.square(self.gradient))
        print("Finished Stochastic Gradient Descent")

    def confusion_metrics(self, conf, save_dir):
        TP = conf[1][1]
        TN = conf[0][0]
        FP = conf[1][0]
        FN = conf[0][1]
        Accuracy = (TP+TN)/(FP+FN+TP+TN) #all pos/all
        print("Accuracy: " + str(Accuracy))


        Recall_1 = TP/(TP+FN)
        Precision_1 = TP/(TP+FP) # True positive/ predicted positives
        print("Metrics from confusion matrics for class name")
        print("Recall: " + str(Recall_1))
        print("Precision: " + str(Precision_1))


        Recall_0 = TN/(TN+FP)
        Precision_0 = TN/(TN + FN)
        print("Metrics from confusion matrics for class no name")
        print("Recall: " + str(Recall_0))
        print("Precision: " + str(Precision_0))
        with open(save_dir +"_Metrics", "w") as file:
            file.write(f"Accuracy : {Accuracy}\n"
            f"For class 0 (different), Recall: {Recall_0}, Precision: {Precision_0}\n"
            f"For class 1 (same), Recall: {Recall_1}, Precision: {Precision_1}"
            )

    def classify_datapoints(self, test_data, test_labels, save_dir):
        """
        Classifies datapoints
        """

        """Saves the model (maybe not the best place) """
        theta_str = ' '.join([str(num) for num in self.theta])
        with codecs.open(save_dir, 'w', 'utf-8') as f:
            f.write(theta_str)
            
        #print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))
        self.DATAPOINTS = len(test_data)
        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))
        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1
        self.confusion_metrics(confusion, save_dir)

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
