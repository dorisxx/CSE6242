from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        self.max_dep = 50

    #find the split that gives teh maximum information gain
    def find_split(self,X,y):
        split_val = X[0][0]
        split_col, max_info_gain = 0, 0
        
        for col in range(len(X[0])):
            unique_col = set([row[col] for row in X])
            for val in unique_col:
                y_left,y_right = partition_classes(X,y,col,val)[2:4]
                ig = information_gain(y,[y_left,y_right])
                if ig>max_info_gain:
                    max_info_gain = ig
                    split_col, split_val = col,val
        return split_col, split_val
            
    def most_y(self,y):
        val, count = np.unique(y, return_counts = True)
        max_index = np.argmax(count)
        return val[max_index]
        
    def only_y(self,y):
        val,count = np.unique(y,return_counts = True)
        if len(val)==1:
            return True
        else:
            return False
        
    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        '''
        build the tree by splitting X until all nodes have a single label.
        '''
        if len(y)==0:
            self.tree['valid'] = 'no' 
           
        self.tree = self.grow_tree(X,y,1)
        
        
    def grow_tree(self, X,y, depth):
        if depth>= self.max_dep:
            return self.most_y(y)
        if self.only_y(y):
            return y[0]
        split_col,split_val = self.find_split(X,y)
        X_left, X_right, y_left, y_right = partition_classes(X,y,split_col,split_val)
        if len(X_left) ==0 or len(X_right)==0:
            return self.most_y(y)
        else:
            d_tree = {}
            d_tree[split_col] = [split_val, self.grow_tree(X_left,y_left,depth+1),self.grow_tree(X_right,y_right,depth+1)]
            return d_tree
            

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        trained = self.tree
        while isinstance(trained,dict):
            split_col = trained.keys()[0]
            split_val = trained[split_col][0]
            if type(split_val) == str:
                if record[split_col] == split_val:
                    trained = trained[split_col][1]
                else:
                    trained = trained[split_col][2]
            else:
                if record[split_col] <= split_val:
                    trained = trained[split_col][1]
                else:
                    trained = trained[split_col][2]
        return trained 
            
            
