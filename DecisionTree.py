#!/usr/bin/env python3

import numpy as np
import random
import pandas as pd
from tqdm import tqdm

import copy
from sklearn.tree import DecisionTreeClassifier

class Node():
    def __init__(self, node_id = 0, left = None, right = None, parent = None, feature = None, threshold = None, value = None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.parent = parent
        self.node_id = node_id

class Tree():
    def __init__(self, max_depth, input_dim, min_dims, max_dims, n_classes, seed):
        np.random.seed(seed)
        random.seed(seed)

        self.depth = max_depth
        
        feature = np.random.randint(low = 0, high = input_dim, size=1)[0]
        threshold = np.random.uniform(min_dims[feature], max_dims[feature])
        self.head = Node(0, None, None, None, feature, threshold, None)
        self.nodes = [self.head]
        
        inner_nodes = [(self.head, 0)]
        while len(inner_nodes) > 0:
            cur_node, cur_depth = inner_nodes.pop()

            feature = np.random.randint(low = 0, high = input_dim, size=1)[0]
            threshold = np.random.uniform(min_dims[feature], max_dims[feature])
            if (cur_depth == max_depth):
                value = np.random.uniform(0, 1, size=n_classes)
                value = value / sum(value)
            else:
                value = None
            left = Node(len(self.nodes), None, None, cur_node, feature, threshold, value)
            cur_node.left = left
            self.nodes.append(left)

            feature = np.random.randint(low = 0, high = input_dim, size=1)[0]
            threshold = np.random.uniform(min_dims[feature], max_dims[feature])
            if (cur_depth == max_depth):
                value = np.random.uniform(0, 1, size=n_classes)
                value = value / sum(value)
            else:
                value = None
            right = Node(len(self.nodes), None, None, cur_node, feature, threshold, value)
            cur_node.right = right

            if cur_depth < max_depth:
                inner_nodes.append((left, cur_depth+1))
                inner_nodes.append((right, cur_depth+1))
            self.nodes.append(right)

        self.lefts = []
        self.rights = []
        self.thresholds = []
        self.values = []
        self.features = []
        for i, node in enumerate(self.nodes):
            if node.left is not None:
                self.lefts.append(node.left.node_id)
            else:
                self.lefts.append(-1)

            if node.right is not None:
                self.rights.append( node.right.node_id )
            else:
                self.rights.append(-1)

            if node.feature is not None:
                self.features.append( node.feature ) 
            else:
                self.features.append(-1)

            if node.threshold is not None:
                self.thresholds.append(node.threshold)
            else:
                self.thresholds.append(-1)

            if node.value is not None:
                self.values.append( np.array([node.value]) ) 
            else:
                self.values.append( np.array( [[-1 for _ in range(n_classes)]])  ) 
        
        self.lefts = np.array(self.lefts)
        self.rights = np.array(self.rights)
        self.thresholds = np.array(self.thresholds)
        self.values = np.array(self.values)
        self.features = np.array(self.features)

    def predict_proba(self, X):
        preds = []
        for x in X:
            cur_node = self.head
            while(cur_node.value is None):
                if x[cur_node.feature] >= cur_node.threshold:
                    cur_node = cur_node.right
                else:
                    cur_node = cur_node.left
            preds.append(cur_node.value)

        return np.array(preds)

#https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy
def gather_numpy(X, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = X.shape[:dim] + X.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                        ", all dimensions of index and X should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(X, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values):
    """
    Common functions used by all tree algorithms to generate the parameters according to the tree_trav strategies.
    Args:
        left: The left nodes
        right: The right nodes
        features: The features used in the decision nodes
        thresholds: The thresholds used in the decision nodes
        values: The values stored in the leaf nodes
    Returns:
        An array containing the extracted parameters
    """
    if len(lefts) == 1:
        # Model creating tree with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        n_classes = values.shape[1] if type(values) is np.ndarray else 1
        values = np.array([np.array([0.0]), values[0], values[0]])
        values.reshape(3, n_classes)

    ids = [i for i in range(len(lefts))]
    nodes = list(zip(ids, lefts, rights, features, thresholds, values))

    # Refactor the tree parameters in the proper format.
    nodes_map = {0: Node(0)}
    current_node = 0
    for i, node in enumerate(nodes):
        id, left, right, feature, threshold, value = node

        if left != -1:
            l_node = Node(left)
            nodes_map[left] = l_node
        else:
            lefts[i] = id
            l_node = -1
            feature = -1

        if right != -1:
            r_node = Node(right)
            nodes_map[right] = r_node
        else:
            rights[i] = id
            r_node = -1
            feature = -1

        nodes_map[current_node].left = l_node
        nodes_map[current_node].right = r_node
        nodes_map[current_node].feature = feature
        nodes_map[current_node].threshold = threshold
        nodes_map[current_node].value = value

        current_node += 1

    lefts = np.array(lefts)
    rights = np.array(rights)
    features = np.array(features)
    thresholds = np.array(thresholds)
    values = np.array(values)

    return [nodes_map, ids, lefts, rights, features, thresholds, values]

def get_parameters_for_tree_trav_sklearn(lefts, rights, features, thresholds, values, num_trees = None):
    """
    This function is used to generate tree parameters for sklearn trees.
    Includes SklearnRandomForestClassifier/Regressor, and SklearnGradientBoostingClassifier.
    Args:
        left: The left nodes
        right: The right nodes
        features: The features used in the decision nodes
        thresholds: The thresholds used in the decision nodes
        values: The values stored in the leaf nodes
    Returns:
        An array containing the extracted parameters
    """
    features = [max(x, 0) for x in features]
    values = np.array(values)
    if len(values.shape) == 3:
        values = values.reshape(values.shape[0], -1)
    if values.shape[1] > 1:
        values /= np.sum(values, axis=1, keepdims=True)
    if num_trees is not None:
        values /= num_trees

    return get_parameters_for_tree_trav_common(lefts, rights, features, thresholds, values)

def from_sklearn(tree_parameters):
    return [
        get_parameters_for_tree_trav_sklearn(
            *tree_param, None
        )
        for tree_param in tree_parameters
    ]

def _find_depth(node, current_depth):
    """
    Recursive function traversing a tree and returning the maximum depth.
    """
    if node.left == -1 and node.right == -1:
        return current_depth + 1
    elif node.left != -1 and node.right == -1:
        return _find_depth(node.l, current_depth + 1)
    elif node.right != -1 and node.left == -1:
        return _find_depth(node.r, current_depth + 1)
    elif node.right != -1 and node.left != -1:
        return max(_find_depth(node.left, current_depth + 1), _find_depth(node.right, current_depth + 1))

def _find_max_depth(tree_parameters):
    """
    Function traversing all trees in sequence and returning the maximum depth.
    """
    depth = 0

    for tree in tree_parameters:
        tree = copy.deepcopy(tree)

        lefts = tree.lefts
        rights = tree.rights

        ids = [i for i in range(len(lefts))]
        nodes = list(zip(ids, lefts, rights))

        nodes_map = {0: Node(0)}
        current_node = 0
        for i, node in enumerate(nodes):
            id, left, right = node

            if left != -1:
                l_node = Node(left)
                nodes_map[left] = l_node
            else:
                lefts[i] = id
                l_node = -1

            if right != -1:
                r_node = Node(right)
                nodes_map[right] = r_node
            else:
                rights[i] = id
                r_node = -1

            nodes_map[current_node].left = l_node
            nodes_map[current_node].right = r_node

            current_node += 1

        depth = max(depth, _find_depth(nodes_map[0], -1))

    return depth

def generate_new_tree(mode, X, Y, max_depth, n_features, n_classes, seed):
    if X is None or Y is None or mode == "random":
        rndtree = Tree(max_depth - 1, n_features, [0 for i in range(n_features)], [1 for i in range(n_features)], n_classes, seed )
        tree_parameters = [
            [rndtree.lefts, rndtree.rights, rndtree.features, rndtree.thresholds, rndtree.values]
        ]
    else:
        if mode == "sklearn":
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        else:
            tree = DecisionTreeClassifier(max_depth=max_depth, splitter="random", max_features = 1, random_state=seed)

        tree.fit(X, Y)
        tree_parameters = [
            [est.tree_.children_left, est.tree_.children_right, est.tree_.feature, est.tree_.threshold, est.tree_.value] for est in [tree]
        ]
    
    return from_sklearn(tree_parameters) 

def _expand_indexes(batch_size, nodes_offset, num_trees):
    indexes = nodes_offset
    indexes = indexes.expand(batch_size, num_trees)
    indexes = np.repeat(indexes, batch_size, axis = 1)

    return indexes.reshape(-1)

def tensor_predict_proba(tree_parameters, X, n_classes):
    num_trees = len(tree_parameters)
    
    max_tree_depth = _find_max_depth(tree_parameters) #max_depth
    num_nodes = max([len(tree_parameter[1]) for tree_parameter in tree_parameters])

    lefts = np.zeros((num_trees, num_nodes), dtype=np.int64)
    rights = np.zeros((num_trees, num_nodes), dtype=np.int64)

    features = np.zeros((num_trees, num_nodes), dtype=np.int64)
    thresholds = np.zeros((num_trees, num_nodes), dtype=np.float32)
    values = np.zeros((num_trees, num_nodes, n_classes), dtype=np.float32)

    for i in range(num_trees):
        lefts[i][: len(tree_parameters[i][0])] = tree_parameters[i][2]
        rights[i][: len(tree_parameters[i][0])] = tree_parameters[i][3]
        features[i][: len(tree_parameters[i][0])] = tree_parameters[i][4]
        thresholds[i][: len(tree_parameters[i][0])] = tree_parameters[i][5]
        values[i][: len(tree_parameters[i][0])][:] = tree_parameters[i][6]

    lefts = lefts.reshape(-1)
    rights = rights.reshape(-1)

    features = features.reshape(-1)
    thresholds = thresholds.reshape(-1)
    values = values.reshape(-1, n_classes)

    nodes_offset = np.array( [[i * num_nodes for i in range(num_trees)]] )
    indexes = _expand_indexes(X.shape[0], nodes_offset, num_trees)

    for _ in range(max_tree_depth):
        tree_nodes = indexes
        feature_nodes = np.take(features, axis=0, indices = tree_nodes).reshape( -1, num_trees )
        feature_values = gather_numpy(X, 1, feature_nodes)

        thresholds = np.take(thresholds, axis=0, indices = indexes).reshape(-1, num_trees)
        lefts = np.take(lefts, axis=0, indices = indexes).reshape(-1, num_trees)
        rights = np.take(rights, axis=0, indices = indexes).reshape(-1, num_trees)
        
        indexes = np.where(feature_values >= thresholds, rights, lefts)
        indexes = indexes + nodes_offset
        indexes = indexes.reshape(-1)

        output = np.take(values, axis = 0, indices = indexes)
        output = output.reshape(-1, num_trees, n_classes)
        output = output.sum(1)
        
        return output

# class TensorTree():
#     """
#     Class implementing the Tree Traversal strategy in PyTorch for tree-base models.
#     """

#     # def __init__(self, max_depth, X, y, seed):
#     def __init__(self, mode, X, Y, max_depth, n_features, n_classes, seed):
#         if X is None or Y is None or mode == "random":
#             rndtree = Tree(max_depth - 1, n_features, [0 for i in range(n_features)], [1 for i in range(n_features)], n_classes, seed )
#             tree_parameters = [
#                 [rndtree.lefts, rndtree.rights, rndtree.features, rndtree.thresholds, rndtree.values]
#             ]
#         else:
#             if mode == "sklearn":
#                 tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
#             else:
#                 tree = DecisionTreeClassifier(max_depth=max_depth, splitter="random", max_features = 1, random_state=seed)

#             tree.fit(X, Y)
#             tree_parameters = [
#                 [est.tree_.children_left, est.tree_.children_right, est.tree_.feature, est.tree_.threshold, est.tree_.value] for est in [tree]
#             ]
        
#         tree_parameters = from_sklearn(tree_parameters) 

#         # self.X_ = X
#         # self.Y = y
#         #print("RNDTREE: ", rndtree.predict_proba(X))
#         # print("RNDTREE: ", tree_parameters)

#         # # print("SKLEARN TREE: ", tree.predict_proba(X))
#         # print("SKLEARN: ", tree_parameters)
#         # Initialize the actual model.
#         self.n_features = n_features
#         self.n_classes = n_classes
#         self.max_tree_depth = _find_max_depth(tree_parameters) #max_depth
#         self.num_trees = len(tree_parameters)
#         self.num_nodes = max([len(tree_parameter[1]) for tree_parameter in tree_parameters])

#         lefts = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
#         rights = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)

#         features = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
#         thresholds = np.zeros((self.num_trees, self.num_nodes), dtype=np.float32)
#         values = np.zeros((self.num_trees, self.num_nodes, self.n_classes), dtype=np.float32)

#         for i in range(self.num_trees):
#             lefts[i][: len(tree_parameters[i][0])] = tree_parameters[i][2]
#             rights[i][: len(tree_parameters[i][0])] = tree_parameters[i][3]
#             features[i][: len(tree_parameters[i][0])] = tree_parameters[i][4]
#             thresholds[i][: len(tree_parameters[i][0])] = tree_parameters[i][5]
#             values[i][: len(tree_parameters[i][0])][:] = tree_parameters[i][6]

#         self.lefts = lefts.reshape(-1)
#         self.rights = rights.reshape(-1)

#         self.features = features.reshape(-1)
#         self.thresholds = thresholds.reshape(-1)
#         self.values = values.reshape(-1, self.n_classes)

#         self.nodes_offset = np.array( [[i * self.num_nodes for i in range(self.num_trees)]] )
        

#     def predict_proba(self, X):
#         indexes = self._expand_indexes(X.shape[0])

#         for _ in range(self.max_tree_depth):
#             tree_nodes = indexes
#             # feature_nodes = torch.index_select(self.features, 0, tree_nodes).view(-1, self.num_trees)
#             # feature_values = torch.gather(X, 1, feature_nodes)
#             feature_nodes = np.take(self.features, axis=0, indices = tree_nodes).reshape( -1, self.num_trees )
#             feature_values = gather_numpy(X, 1, feature_nodes)

#             thresholds = np.take(self.thresholds, axis=0, indices = indexes).reshape(-1, self.num_trees)
#             lefts = np.take(self.lefts, axis=0, indices = indexes).reshape(-1, self.num_trees)
#             rights = np.take(self.rights, axis=0, indices = indexes).reshape(-1, self.num_trees)
#             # thresholds = torch.index_select(self.thresholds, 0, indexes).view(-1, self.num_trees)
#             # lefts = torch.index_select(self.lefts, 0, indexes).view(-1, self.num_trees)
#             # rights = torch.index_select(self.rights, 0, indexes).view(-1, self.num_trees)
            
#             indexes = np.where(feature_values >= thresholds, rights, lefts)
#             # indexes = torch.where(torch.ge(feature_values, thresholds), rights, lefts).long()
#             indexes = indexes + self.nodes_offset
#             indexes = indexes.reshape(-1)

#         # output = torch.index_select(self.values, 0, indexes).view(-1, self.num_trees, self.n_classes)
#         # print(self.values)
#         # print(self.lefts)
#         # print(self.rights)
#         # print(self.n_classes * indexes)
#         output = np.take(self.values, axis = 0, indices = indexes)
#         output = output.reshape(-1, self.num_trees, self.n_classes)
#         output = output.sum(1)
        
#         return output

        # if self.regression:
        #     return output

        # if self.anomaly_detection:
        #     # Select the class (-1 if negative) and return the score.
        #     return torch.where(output.view(-1) < 0, self.classes[0], self.classes[1]), output

        # if self.perform_class_select:
        #     return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        # else:
        #     return torch.argmax(output, dim=1), output


# X = np.array([
#     [1,1,1],
#     [0,0,0],
#     [0.5, 0.5, 0.5],
#     [0.25, 0.25, 0.25],
#     [0.75, 0.75, 0.75]
# ])

# Y = np.array([
#     1,
#     0,
#     1,
#     1,
#     0
# ])

# # tree = TensorTree(2, 3, [0,0,0], [1,1,1], 2, 1234)
# tree = TensorTree(2,X,Y,1234)
# pred = tree.predict_proba(X)
# print("TENSOR TREE ", pred)