import sys
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array(np.split(X_train,num_folds))
y_train_folds = np.array(np.split(y_train,num_folds))
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################
# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = np.zeros(num_folds)
print k_to_accuracies
################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
folds = range(num_folds);   
for f in xrange(num_folds):    
    # Create the training set. We leave out fold f for testing
    indices = np.array(list(set(folds) - set([f])))

    training_folds_x = np.concatenate(X_train_folds[indices]);
    training_folds_y = np.concatenate(y_train_folds[indices]);

    test_fold_x = X_train_folds[f];
    test_fold_y = y_train_folds[f];
    num_test = test_fold_x.shape[0];
    print "Fold: ", f; #, " training sizes: ", training_folds_x.shape, training_folds_y.shape, " testing sizes: ",
                #test_fold_x.shape, test_fold_y.shape
    sys.stdout.flush();

    # Train the classifier using the 4 training folds
    classifier = KNearestNeighbor()
    classifier.train(training_folds_x, training_folds_y)

    # Test the clasifier using the test fold
    dists = classifier.compute_distances_no_loops(test_fold_x)
    #print "Computed distances: ", dists.shape;
    #sys.stdout.flush();
    for k in k_choices:
        print "Testing k is: ", k;
        sys.stdout.flush();
        y_test_pred = classifier.predict_labels(dists, k=k)
        #print "Fold: ", f, " predictions: ", y_test_pred.shape, " actuals: ", test_fold_y.shape
        #sys.stdout.flush();

        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(y_test_pred == test_fold_y)
        k_to_accuracies[k][f] = float(num_correct) / num_test
        #print "Fold: ", f, " Accuracy: ", accuracy[f];
        #sys.stdout.flush();
    print "Found accuracies: ", k_to_accuracies
    sys.stdout.flush();
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    #for accuracy in k_to_accuracies[k]:
    print 'k = %d, accuracy = %f' % (k, np.mean(k_to_accuracies[k]))
