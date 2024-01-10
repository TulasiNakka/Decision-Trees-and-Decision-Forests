import math
import random

dataset = []
count_att = 0
maxClass = 0

class Node(object):
    def __init__(self, treeId, nodeId, attribute, threshold, gain):
        self.treeId = treeId
        self.nodeId = nodeId
        self.featureId = attribute
        self.threshold = threshold
        self.gain = gain
        self.leftChild = None
        self.rightChild = None

    def eval(self, sample):
        if sample[self.featureId] < self.threshold:
            if isinstance(self.leftChild, Node):
                return self.leftChild.eval(sample)
            else:
                return self.leftChild
        else:
            if isinstance(self.rightChild, Node):
                return self.rightChild.eval(sample)
            else:
                return self.rightChild
            

    def printTree(self):
        queue = [self]
        while queue:
            currentNode = queue.pop(0)
            print(f"tree={currentNode.treeId}, node={currentNode.nodeId}, feature={currentNode.featureId}, thr={currentNode.threshold}, gain={currentNode.gain}\n")
            if isinstance(currentNode.leftChild, Node):
                queue.append(currentNode.leftChild)
            if isinstance(currentNode.rightChild, Node):
                queue.append(currentNode.rightChild)

def CHOOSE_ATTRIBUTE(examples, attributes):
    max_gain = -1
    best_attribute = -1
    best_threshold = -1
    for attribute in attributes:
        L, M = min_max_value(examples, attribute)
        for K in range(1, 51):
            threshold = L + (K * (M - L) / 51)
            gain = INFORMATION_GAIN(examples, attribute, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
                best_threshold = threshold
    return best_attribute, best_threshold, max_gain

def choose_random_attribute(examples, attributes):
    max_gain = -1
    best_attribute = -1
    best_threshold = -1
    A = random.choice(attributes)
    L, M = min_max_value(examples, A)
    for K in range(1, 51):
        threshold = L + (K * (M - L) / 51)
        gain = INFORMATION_GAIN(examples, A, threshold)
        if gain > max_gain:
            max_gain = gain
            best_attribute = A
            best_threshold = threshold
    return best_attribute, best_threshold, max_gain

def INFORMATION_GAIN(examples, A, threshold):
    initial_dist = count_examples(examples)
    left, right = splitBy(examples, A, threshold)
    left_dist = count_examples(left)
    right_dist = count_examples(right)
    left_weight = float(sum(left_dist)) / sum(initial_dist)
    right_weight = float(sum(right_dist)) / sum(initial_dist)
    info_gain = entropy(initial_dist) - (left_weight * entropy(left_dist)) - (right_weight * entropy(right_dist))
    return info_gain

def entropy(DISTRIBUTION):
    ent = 0
    temp=0
    total = sum(DISTRIBUTION)
    for dist in DISTRIBUTION:
        if total:
            temp = float(dist) / total
        if not temp == 0.0000:
            ent = ent - (temp * math.log(temp, 2))
    return ent

def splitBy(examples, A, threshold):
    left = []
    right = []
    for example in examples:
        if example["attr"][A] < threshold:
            left.append(example)
        else:
            right.append(example)
    return left, right

def count_examples(examples):
    global maxClass
    c_count = [0] * (maxClass + 1)
    for ex in examples:
        c_count[ex['class']] = c_count[ex["class"]] + 1
    return c_count

def DISTRIBUTION(examples):
    c_count = count_examples(examples)
    DISTRIBUTION = []
    length = len(c_count)
    length = len(c_count)
    DISTRIBUTION = [float(c) / length for c in c_count]
    return DISTRIBUTION
    
def min_max_value(examples, A):
    Min = 99999.99
    Max = 0.0
    for example in examples:
        if example["attr"][A] < Min:
            Min = example["attr"][A]
        if example["attr"][A] > Max:
            Max = example["attr"][A]
    return Min, Max

def same_class_value(examples):
    count = count_examples(examples)
    numOfZeros = 0
    for c in count:
        if c == 0:
            numOfZeros = numOfZeros + 1
    return numOfZeros == len(count) - 1

# Decision Tree Learning
def DTL(treeId, examples, attributes, default, nodeId, options):
    if not examples:
        return default
    # Same class examples
    elif same_class_value(examples): 
        return DISTRIBUTION(examples)
    else:
        if options == "optimized":
            best_attribute, best_threshold, max_gain = CHOOSE_ATTRIBUTE(examples, attributes)
        else:
            best_attribute, best_threshold, max_gain = choose_random_attribute(examples, attributes)
        tree = Node(treeId, nodeId, best_attribute, best_threshold, max_gain)
        examples_left, examples_right = splitBy(examples, best_attribute, best_threshold)

        # Pruning
        if len(examples_left) > 50: 
            leftsubtree = DTL(treeId, examples_left, attributes, DISTRIBUTION(examples), 2 * nodeId, options)

        else:
            leftsubtree = DISTRIBUTION(examples_left)
        if len(examples_right) > 50:
            rightsubtree = DTL(treeId, examples_right, attributes, DISTRIBUTION(examples), (2 * nodeId) + 1, options)
        else:
            rightsubtree = DISTRIBUTION(examples_right)
        tree.leftChild = leftsubtree
        tree.rightChild = rightsubtree
        return tree


def example_attributes(examples, attribute, isLess, threshold):
    new_examples = []
    for example in examples:
        if isLess and example[attribute] < threshold:
            new_examples.append(example)
        elif not isLess and example[attribute] >= threshold:
            new_examples.append(example)
    return new_examples

def evaluate(forest, sample):
    results = []
    for tree in forest:
        results.append(tree.eval(sample))
    avg = results[0]
    iterResults = iter(results)
    next(iterResults)
    for result in iterResults:
        avg = [sum(dist) / len(forest) for dist in zip(*results)]
    avg = [val / len(results) for val in avg]    
    return avg

from sys import argv
if __name__ == "__main__":
    if len(argv) == 4:
        script, training_file_name, test_file_name, options = argv
        if options == 'optimized' or options == 'randomized':
            forest_length = 1
        elif options.startswith('forest'):
            forest_length = int(options[6:])
            options = 'randomized'
        else:
            print("Invalid option. Please use 'optimized', 'randomized', 'forest3', or 'forest15'.")
            raise SystemExit
    else:
        print("Usage: dtree <training_file> <test_file> <optimized/randomized/forestN>")
        raise SystemExit
   
    # Training phase
    training_file = open(training_file_name)
    count_att = 0 
    for sample in training_file:
        split = sample.split()
        row = []
        count_att = len(split) - 1
        for attribute in split[:-1]:
            row.append(float(attribute))
        label = int(split[-1])
        if label > maxClass:
            maxClass = label
            dataset.append({"attr": row, "class": label})

    if options[:3] == 'forest':
        forest_length = int(options[6:])
        options = 'randomized'
    forest = []
    for i in range(0, forest_length):
        forest.append(DTL(i, dataset, range(0, count_att), DISTRIBUTION(dataset), 1, options))
    for tree in forest:
        tree.printTree()

    # Testing phase
    test_file = open(test_file_name)
    accuracies = []
    for i, sample in enumerate(test_file):
        split = sample.split()
        row = []
        for attribute in split[:-1]:
            row.append(float(attribute))
        prob_dist = evaluate(forest, row)  
        prob_dist_set = list(set(prob_dist))
        predicted_class = prob_dist.index(max(prob_dist_set))

        if len(set(prob_dist)) == 1:
            # No ties
            accuracy = 1.0 if predicted_class == int(split[-1]) else 0.0
        else:
            # Ties
            if predicted_class == int(split[-1]) and int(split[-1]) in [i for i, prob in enumerate(prob_dist) if prob == max(prob_dist)]:
                accuracy = 1.0 / prob_dist.count(max(prob_dist))
            else:
                accuracy = 0.0

        accuracies.append(accuracy)
        print("Index=%5d, Result=%3d, True Class=%3d, Accuracy=%4.2lf\n" % (i, predicted_class, int(split[-1]), accuracy))

    accuracy_percentage = ((sum(accuracies) / len(accuracies)))
print(f"Classification Accuracy={accuracy_percentage:.2f}\n")


