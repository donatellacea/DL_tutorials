def check_task_1(n_layer, n_neurons):
    if (n_layer == 1) & (n_neurons == 1):
        print('Correct solution! You can continue with task 2!')
    elif (n_layer is None) | (n_neurons is None):
        print('Please insert both the number of layers and the total number of neurons.')
    elif (0 == n_layer) | (0 == n_neurons):
        print('Incorrect solution, try again! Try to increase the number of layers or the total number of neurons')
    else:
        print('Incorrect solution, try again! Try to reduce the total number of neurons or the number of layers,'
              'the neural network can still learn fast and correct with less layers.')


def check_task_2(n_layer, n_neurons):
    if (n_layer == 1) & (n_neurons == 3):
        print('Correct solution! You can continue with tasks 3!')
    elif (n_layer is None) | (n_neurons is None):
        print('Please insert both the number of layers and the total number of neurons.')
    elif (n_layer == 0) | (n_neurons == 0):
        print('Incorrect solution, try again! Try to increase the total number of layers or the number of neurons')
    else:
        print('Incorrect solution, try again! Try to increase the total number of neurons or the number of  layers,'
              ' the neural network can still learn fast and correct with less layers.')


def check_task_3():
    print('At the beginning of the training process, the weights are randomly initialized.')
    print('So in the first run the network was starting learning from scratch. ')
    print('In the second run, instead, the network started with a proper initialization,'
          ' because it learnt something from the previous experience.')
    print('In certain cases it might be useful to pre-train the model, i.e. you can use the weights you saved'
          ' from a previous network as the initial weight values for your new experiment.')
    print('Using pre-trained model is a kind of transfer learning, and can help improving the performances '
          'of the model.')


def check_task_4(n_layer, n_neurons):
    if (n_layer is None) | (n_neurons is None):
        print('Please insert both the number of layers and total the number of neurons.')
    elif 6 > n_layer:
        print('Incorrect solution! Try to increase the number of layers or you cannot rach the desired performances!')
    else:
        if 37 < n_neurons:
            print('Incorrect solution! Try to reduce the number of total neurons, you can still find reach the '
                  'desired performances with less neurons.')
        elif 37 > n_neurons:
            print('Incorrect solution! Try to increase the number of total neurons'
                  ' or you cannot reach the desired performances!')
        else:
            print('Correct solution! Congratulations, you completed the challenge!')


def check_task_tm_2_1():
    print('When the training curve is decreasing as expected, but the test loss is increasing, it means that the '
          'model is not generalizing well on the data. This is a common problem in data science called overfitting.'
          'There are several approaches to reduce it, one might be to have more training data, since so far the model'
          "has too many parameters and it's learning perfectly to reproduce the training set, but is not able to "
          "generalize this result on new data.")


def check_task_tm_2_2(confusion_matrix, sensitivity, specificity):
    if None in confusion_matrix:
        print('Please, insert all the values in the confusion matrix, and run the cell again.')
        return None
    else:
        tp = confusion_matrix[0]
        fn = confusion_matrix[1]
        fp = confusion_matrix[2]
        tn = confusion_matrix[3]
        sens = round(tp / (tp + fn), 2)
        spec = round(tn / (fp + tn), 2)

        if sensitivity == sens and specificity == spec:
            print('Correct solution! This model is still not perfect. What could we do to improve it?')
        elif sensitivity != sens and specificity != spec:
            print('Incorrect solution! Please remember that the sensitivity is the true positive rate '
                  'and the specificity is the true negative rate.')
        elif sensitivity != sens and specificity == spec:
            print('Incorrect solution! Please remember that the sensitivity is the true positive rate.')
        elif sensitivity == sens and specificity != spec:
            print('Incorrect solution! Please remember that the specificity is the true negative rate.')
        elif sensitivity is None or specificity is None:
            print('Please, insert a value for both sensitivity and specificity.')
