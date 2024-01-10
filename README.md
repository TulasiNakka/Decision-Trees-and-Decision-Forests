Implement decision trees and decision forests. Your program will learn decision trees from training data and will apply decision trees and decision forests to classify test objects.

Command-line Arguments
Program learns a decision tree for a binary classification problem, given some training data. In particular, your program will run as follows:
dtree training_file test_file option
The arguments provide to the program the following information:
The first argument is the name of the training file, where the training data is stored.
The second argument is the name of the test file, where the test data is stored.
The third argument can have four possible values: optimized, randomized, forest3, or forest15. It specifies how to train (learn) the decision tree, and will be discussed later.
Both the training file and the test file are text files, containing data in tabular format. Each value is a number, and values are separated by white space. The i-th row and j-th column contain the value for the j-th feature of the i-th object. The only exception is the LAST column, that stores the class label for each object. Make sure you do not use data from the last column (i.e., the class labels) as attributes (features) in your decision tree.
Example files that can be passed as command-line arguments are in the datasets directory. That directory contains three datasets, copied from the UCI repository of machine learning datasets:

For each dataset, a training file and a test file are provided. The name of each file indicates what dataset the file belongs to, and whether the file contains training or test data.
Note that, for the purposes of your assignment, it does not matter at all where the data came from. One of the attractive properties of decision trees (and many other machine learning methods) is that they can be applied in the exact same way to many different types of data, and produce useful results.

Training Phase
The first thing that your program should do is a decision tree using the training data. What you train and how you do the training depends on the came of the third command line argument, that we called "option". This option can take four possible values, as follows:
optimized: in this case, at each non-leaf node of the tree (starting at the root) you should identify the optimal combination of attribute (feature) and threshold, i.e., the combination that leads to the highest information gain for that node.
randomized: in this case, at each non-leaf node of the tree (starting at the root) you should choose the attribute (feature) randomly. The threshold should still be optimized, i.e., it should be chosen so as to maximize the information gain for that node and for that randomly chosen attribute.
forest3: in this case, your program trains a random forest containing three trees. Each of those trees should be trained as discussed under the "randomized" option.
forest15: in this case, your program trains a random forest containing 15 trees. Each of those trees should be trained as discussed under the "randomized" option.
All four options are described in more details in the lecture slides titled Practical Issues with Decision Trees. Your program should follow the guidelines stated in those slides.

Testing Phase
For each test object (each line in the test file) print out, in a separate line, the classification label. If your classification result is a tie among two or more classes, choose one of them randomly. For each test object you should print a line containing the following info:
index. This is the index of the object (the line where it occurs) in the test file. Start with 0 in numbering the objects, not with 1.
predicted class (the result of the classification).
true class (from the last column of the test file).
accuracy. This is defined as follows:
 If there were no ties in your classification result, and the predicted class is correct, the accuracy is 1.
 If there were no ties in your classification result, and the predicted class is incorrect, the accuracy is 0.
 If there were ties in your classification result, and the correct class was one of the classes that tied for best, the accuracy is 1 divided by the number of classes that tied for best.
 If there were ties in your classification result, and the correct class was NOT one of the classes that tied for best, the accuracy is 0.
Use the following Format for each line: Object Index = <index>, Result = <predicted class>, True Class = <true class>, Accuracy = <accuracy>

After you have printed the results for all test objects, you should print the overall classification accuracy, which is defined as the average of the classification accuracies you printed out for each test object.

Use the following Format for this: Classification Accuracy = <average of all accuracies>


