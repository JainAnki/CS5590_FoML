# FoML Assign 1 Code Skeleton
# # Please use this outline to implement your decision tree. You can add any code around this.

from collections import Counter # Used for counting 
import random
import csv
from math import log # We need this to compute log base 2 

# Enter You Name Here
myname = "Ankita Jain" # or "Amar-Akbar-Antony"

# Implement your decision tree below
class Node:   
    # Method used to initialize a new node's data fields with initial values
    def __init__(self, target):
    
        # Declaring variables specific to this node
        self.feature = None  # Input feature
        self.feature_values = []  # Values of the feature 
        self.target = target   # Target class for the node
        self.children = {}   # To Keep track of the node's children
        
        # References to the parent node
        self.root_feature = None
        self.root_feature_value = None

        # Used for pruned trees
        self.pruned = False   
        self.instances_classified = []


class DecisionTree():

    def learn(self,data, default, Impurity):
        # The len method returns the number of items in the list
        # If there are no more instances left, return a leaf that is labeled with 
        # the default class
        if len(data) == 0:
            return Node(default)
    
        targets = []  # Create an empty list named 'classes'
        for instance in data:
            targets.append(instance['quality'])
    
        if len(Counter(targets)) == 1 or len(targets) == 1:
            tree = Node(self.most_common_class(data))
            return tree
    
        # Otherwise, find the best attribute, the attribute that maximizes the gain 
        # ratio of the data set, to be the next decision node.
        else:
            best_feature = self.feature_to_split(data, Impurity)
    
            # Initialize a tree with the most common class
            tree = Node(self.most_common_class(data))
    
            # The ost informative attribute becomes this decision node
            # e.g. "Outlook" becomes this node
            tree.feature = best_feature
    
            best_feature_values = []
    
            # The branches of the node are the values of the best_attribute
            # e.g. "Sunny", "Overcast", "Rainy"
            # Go through each instance and create a list of the values of 
            # best_attribute
            for instance in data:
                best_feature_values.append(instance[best_feature])
            # Create a list of the unique best attribute values
            tree.feature_values = list(set(best_feature_values))
    
            # Now we need to split the instances. We will create separate subsets
            # for each best attribute value. These become the child nodes
            for best_feat_value_i in tree.feature_values:
    
                # Generate the subset of instances
                instances_i = []
                for instance in data:
                    if instance[best_feature] == best_feat_value_i:
                        instances_i.append(instance) #Add this instance to the list
    
                # Create a subtree 
                subtree = self.learn(instances_i, self.most_common_class(data), Impurity)
    
                # Initialize the values of the subtree
                subtree.instances_classified = instances_i
    
                # Keep track of the state of the subtree's parent (i.e. tree)
                subtree.root_feature = best_feature # parent node
                subtree.root_feature_value = best_feat_value_i # branch name
    
                # Assign the subtree to the appropriate branch
                tree.children[best_feat_value_i] = subtree
    
            # Return the decision tree
            return tree

    def most_common_class(self,data):
        classes = []  # Create an empty list named 'classes'
    
        # For each instance in the list of instances, append the value of the class
        # to the end of the classes list
        for instance in data:
            classes.append(instance['quality'])
    
        # The 1 ensures that we get the top most common class
        # The [0][0] ensures we get the name of the class label and not the tally
        # Return the name of the most common class of the instances
        return Counter(classes).most_common(1)[0][0]
    
    def prior_entropy(self,data, Impurity):
        """
        Calculate the entropy of the data set with respect to the actual class
        prior to splitting the data.
        """
        classes = []  # Create an empty list named 'classes'
    
        for instance in data:
            classes.append(instance['quality'])
        counter = Counter(classes)
    
        # If all instances have the same class, the entropy is 0
        if len(counter) == 1:
            return 0
        else:
        # Compute the weighted sum of the logs of the probabilities of each 
        # possible outcome 
            if Impurity == 0:
                entropy = 0
                for cl, count_of_cl in counter.items():
                    probability = count_of_cl / len(classes)
                    entropy += probability * (log(probability, 2))
                return -entropy
            elif Impurity == 1:
                GiniIndex = 0
                for cl, count_of_cl in counter.items():
                    probability = count_of_cl / len(classes)
                    GiniIndex += probability ** 2
                return (1-GiniIndex)
    
    def entropy(self,data, feature, feature_value, Impurity):
        """
        Calculate the entropy for a subset of the data filtered by attribute value
        """
        classes = []  # Create an empty list named 'classes'
    
        for instance in data:
            if instance[feature] == feature_value:
                classes.append(instance['quality'])
        counter = Counter(classes)
    
        # If all instances have the same class, the entropy is 0
        if len(counter) == 1:
            return 0
        else:
        # Compute the weighted sum of the logs of the probabilities of each 
        # possible outcome 
            if Impurity == 0:
                entropy = 0
                for cl, count_of_cl in counter.items():
                    probability = count_of_cl / len(classes)
                    entropy += probability * (log(probability, 2))
                return -entropy
            elif Impurity == 1:
                GiniIndex = 0
                for cl, count_of_cl in counter.items():
                    probability = count_of_cl / len(classes)
                    GiniIndex += probability ** 2
                return (1-GiniIndex)
    
    def info_gain(self,data, feature, Impurity):
        # Record the entropy of the combined set of instances
        priorentropy = self.prior_entropy(data, Impurity)
    
        values = []
    
        # Create a list of the attribute values for each instance
        for instance in data:
            values.append(instance[feature])
        counter = Counter(values) # Store the frequency counts of each attribute value
    
        # The remaining entropy if we were to split the instances based on this attribute
        # This is a weighted entropy score sum
        remaining_entropy = 0
    
        # items() method returns a list of all dictionary key-value pairs
        for feature_value, feature_value_count in counter.items():
            probability = feature_value_count/len(values)
            remaining_entropy += (probability * self.entropy(
                data, feature, feature_value, Impurity))
    
        information_gain = priorentropy - remaining_entropy
        return information_gain
    
    def feature_to_split(self,data, Impurity):
        """
        Choose the attribute that provides the most information if you were to
        split the data set based on that attribute's values. This attribute is the 
        one that has the highest gain ratio.
        """
        selected_feature = None
        max_gain = -1000
    
        # instances[0].items() extracts the first instance in instances
        # for key, value iterates through each key-value pair in the first
        # instance in instances
        # In short, this code creates a list of the attribute names
        features = [key for key, value in data[0].items()]
        # Remove the "Class" attribute name from the list
        features.remove('quality')
    
        # For every attribute in the list of attributes
        for feature in features:
            # Calculate the gain ratio and store that value
            gain = self.info_gain(data, feature, Impurity)
    
            # If we have a new most informative attribute
            if gain > max_gain:
                max_gain = gain
                selected_feature = feature
    
        return selected_feature

    def get_ten_folds(self,data):
        """
        Parameters:
            instances: A list of dictionaries where each dictionary is an instance. 
                Each dictionary contains attribute:value pairs 
        Returns: 
            fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9
            Ten-Fold Stratified Cross Validation
        """
        # Create ten empty folds
        fold0 = []
        fold1 = []
        fold2 = []
        fold3 = []
        fold4 = []
        fold5 = []
        fold6 = []
        fold7 = []
        fold8 = []
        fold9 = []
    
        # Shuffle the data randomly
        random.shuffle(data)
    
        # Generate a list of the unique class values and their counts
        classes = []  # Create an empty list named 'classes'
    
        # For each instance in the list of instances, append the value of the class
        # to the end of the classes list
        for instance in data:
            classes.append(instance['quality'])
    
        # Create a list of the unique classes
        unique_classes = list(Counter(classes).keys())
    
        # For each unique class in the unique class list
        for uniqueclass in unique_classes:
    
            # Initialize the counter to 0
            counter = 0
            
            for instance in data:
    
                # If we have a match
                if uniqueclass == instance['quality']:
    
                    # Allocate instance to fold0
                    if counter == 0:
    
                        # Append this instance to the fold
                        fold0.append(instance)
    
                        # Increase the counter by 1
                        counter += 1
    
                    # Allocate instance to fold1
                    elif counter == 1:
    
                        # Append this instance to the fold
                        fold1.append(instance)
    
                        # Increase the counter by 1
                        counter += 1
    
                    # Allocate instance to fold2
                    elif counter == 2:
    
                        # Append this instance to the fold
                        fold2.append(instance)
    
                        # Increase the counter by 1
                        counter += 1
    
                    # Allocate instance to fold3
                    elif counter == 3:
    
                        # Append this instance to the fold
                        fold3.append(instance)
    
                        # Increase the counter by 1
                        counter += 1

                    # Allocate instance to fold4
                    elif counter == 4:
    
                        # Append this instance to the fold
                        fold4.append(instance)
    
                        # Increase the counter by 1
                        counter += 1

                    # Allocate instance to fold5
                    elif counter == 5:
    
                        # Append this instance to the fold
                        fold5.append(instance)
    
                        # Increase the counter by 1
                        counter += 1
                    
                    # Allocate instance to fold6
                    elif counter == 6:
    
                        # Append this instance to the fold
                        fold6.append(instance)
    
                        # Increase the counter by 1
                        counter += 1

                    # Allocate instance to fold7
                    elif counter == 7:
    
                        # Append this instance to the fold
                        fold7.append(instance)
    
                        # Increase the counter by 1
                        counter += 1

                    # Allocate instance to fold8
                    elif counter == 8:
    
                        # Append this instance to the fold
                        fold8.append(instance)
    
                        # Increase the counter by 1
                        counter += 1
    
                    # Allocate instance to fold9
                    else:
    
                        # Append this instance to the fold
                        fold9.append(instance)
    
                        # Reset the counter to 0
                        counter = 0
    
        # Shuffle the folds
        random.shuffle(fold0)
        random.shuffle(fold1)
        random.shuffle(fold2)
        random.shuffle(fold3)
        random.shuffle(fold4)
        random.shuffle(fold5)
        random.shuffle(fold6)
        random.shuffle(fold7)
        random.shuffle(fold8)
        random.shuffle(fold9)
        
        # Return the folds
        return  fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9

    def acc(self,trained_tree, test):
        # Set the counter to 0
        no_of_correct_predictions = 0
    
        for test_instance in test:
            if self.classify(trained_tree, test_instance) == test_instance['quality']:
                no_of_correct_predictions += 1
    
        return no_of_correct_predictions / len(test)
    
    def classify(self,node, test_instance):
        '''
        Parameters:
            node: A trained tree node
            test_instance: A single test instance
        Returns:
            Class value (e.g. "Play")
        '''
        # Stopping case for the recursive call.
        # If this is a leaf node (i.e. has no children)
        if len( node.children) == 0:
            return node.target
        # Otherwise, we are not yet on a leaf node.
        # Call predict method recursively until we get to a leaf node.
        else:
            # Extract the attribute name (e.g. "Outlook") from the node. 
            # Record the value of the attribute for this test instance into 
            # attribute_value (e.g. "Sunny")
            feature_value = test_instance[node.feature]
    
            # Follow the branch for this attribute value assuming we have 
            # an unpruned tree.
            
            if feature_value in node.children and node.children[
                feature_value].pruned == False:
                # Recursive call
                return self.classify(node.children[feature_value], test_instance)

            # Otherwise, return the most common class
            # return the mode label of examples with other attribute values for the current attribute
            else:
                instances = []
                for feat_value in node.feature_values:
                    instances += node.children[feat_value].instances_classified
                return self.most_common_class(instances)

    TREE = None
    def prune(self,node, instances):
        """
        Prune the tree recursively, starting from the leaves
        Parameters:
            node: A tree that has already been trained
            val_instances: The validation set        
        """
        global TREE
        TREE = node
        
        def prune_node(node, instances):
            # If this is a leaf node
            if len(node.children) == 0:
                accuracy_before_pruning = self.acc(TREE, instances)
                node.pruned = True

                # If no improvement in accuracy, no pruning
                if accuracy_before_pruning >= self.acc(TREE, instances):
                    node.pruned = False
                return

            for value, child_node in node.children.items():
                prune_node(child_node, instances)

            # Prune when we reach the end of the recursion
            accuracy_before_pruning = self.acc(TREE, instances)
            node.pruned = True

            if accuracy_before_pruning >= self.acc(TREE, instances):
                node.pruned = False

        prune_node(TREE, instances)

def run_decision_tree():

    # Load data set
    data = []
    with open("wine-dataset.csv") as f:
        csv_file = csv.reader(f)
        headers = next(csv_file)

        for row in csv_file:
            data.append(dict(zip(headers, row)))
    print ("Number of records: %d" % len(data))
    random.shuffle(data)

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [data[i] for i in range(len(data)) if i % K != 9]
    test_set = [data[i] for i in range(len(data)) if i % K == 9]

    tree = DecisionTree()
    # Get the most common class in the data set.
    default = tree.most_common_class(data)
    # Construct a tree using training set
    DTree = tree.learn( training_set,default, Impurity = 0 )
    accuracy = tree.acc(DTree, test_set)
    print ("Without cross validation: accuracy = %.4f" % accuracy)       
    
    #Gini Impurity
    DTree = tree.learn( training_set,default, Impurity = 1 )
    accuracy_gini = tree.acc(DTree, test_set)
    print ("Without cross validation Gini Index accuracy = %.4f" % accuracy_gini)       

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("Without cross validation with Entropy as Impurity, Accuracy = %.4f" % accuracy)
    f.write("\n\n"+"Without cross validation with Gini Index as impurity, Accuracy = %.4f" % accuracy_gini)
    # Generate the ten stratified folds
    fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9 = tree.get_ten_folds(
        training_set)
 
    # Generate lists to hold the training and test sets for each experiment
    testset = []
    trainset = []
 
    # Create the training and test sets for each experiment
    # Experiment 0
    testset.append(fold0)
    trainset.append(fold1 + fold2 + fold3 + fold4+ fold5 + fold6 + fold7 + fold8 + fold9)
 
    # Experiment 1
    testset.append(fold1)
    trainset.append(fold0 + fold2 + fold3 + fold4+ fold5 + fold6 + fold7 + fold8 + fold9)
 
    # Experiment 2
    testset.append(fold2)
    trainset.append(fold0 + fold1 + fold3 + fold4+ fold5 + fold6 + fold7 + fold8 + fold9)
 
    # Experiment 3
    testset.append(fold3)
    trainset.append(fold0 + fold1 + fold2 + fold4 + fold5 + fold6 + fold7 + fold8 + fold9)
     
    # Experiment 4
    testset.append(fold4)
    trainset.append(fold0 + fold1 + fold2 + fold3 + fold5 + fold6 + fold7 + fold8 + fold9)

    # Experiment 5
    testset.append(fold5)
    trainset.append(fold1 + fold2 + fold3 + fold4 + fold4 + fold6 + fold7 + fold8 + fold9)
 
    # Experiment 6
    testset.append(fold6)
    trainset.append(fold0 + fold2 + fold3 + fold4 + fold4 + fold5 + fold7 + fold8 + fold9)
 
    # Experiment 7
    testset.append(fold7)
    trainset.append(fold0 + fold1 + fold3 + fold4 + fold4 + fold5 + fold6 + fold8 + fold9)
 
    # Experiment 8
    testset.append(fold8)
    trainset.append(fold0 + fold1 + fold2 + fold4 + fold4 + fold5 + fold6 + fold7 + fold9)
     
    # Experiment 9
    testset.append(fold9)
    trainset.append(fold0 + fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8)
    
    length = len(data)

    pruned_accuracies = []
    unpruned_accuracies = []

    # Run all 10 experiments for 10-fold stratified cross-validation
    for experiment in range(10):
        # Each experiment has a training and testing set that have been 
        # preassigned.
        train = trainset[experiment][:length]
        test = testset[experiment]            
        # Unpruned
        DTree = tree.learn( train,default,Impurity = 1 )
        accuracy = tree.acc(DTree, test)
        print ("With cross validation: Experiment = %0.0f " % experiment)       
        print("Unpruned Tree Accuracy with Gini Index as impurity = %.4f" % accuracy)
        unpruned_accuracies.append(accuracy) 
        
        # Pruned
        DTree = tree.learn( train,default, Impurity = 0 )
        tree.prune(DTree, test_set)
        accuracy = tree.acc(DTree, test)
        pruned_accuracies.append(accuracy)
        print("Pruned Tree Accuracy = %.4f" % accuracy)

        
    # Calculate and store the average classification 
    # accuracies for each experiment
    avg_pruned_accuracies = sum(pruned_accuracies) / len(pruned_accuracies)    
    avg_unpruned_accuracies = sum(unpruned_accuracies) / len(unpruned_accuracies)
    f.write("\n\n" +"Average unpruned tree accuracy with Gini Index Impurity = %.4f" % avg_unpruned_accuracies)
    f.write("\n\n"+"Average pruned tree accuracy with Entropy as Impurity: %0.4f" %avg_pruned_accuracies) 
    f.close()


if __name__ == "__main__":
    run_decision_tree()
