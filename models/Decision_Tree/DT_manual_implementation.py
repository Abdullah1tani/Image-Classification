import numpy as np

class SimpleDecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, features, labels):
        # Combine features and labels into one dataset for processing
        dataset = np.column_stack((features, labels))
        self.tree = self._build_tree(dataset, depth=0)

    def _build_tree(self, dataset, depth):
        # Recursive tree-building function
        node = self._best_split(dataset)
        left, right = node['groups']
        del(node['groups'])

        # Check if we should make this node a terminal node
        if depth >= self.max_depth or not left.size or not right.size:
            node['left'] = node['right'] = self._to_terminal(np.vstack((left, right)))
            return node

        # Build left and right branches recursively
        node['left'] = self._build_tree(left, depth + 1)
        node['right'] = self._build_tree(right, depth + 1)
        return node

    def _best_split(self, dataset):
        # Determine the best split based on Gini index
        class_values = np.unique(dataset[:, -1])
        best_index, best_threshold, best_score, best_groups = None, None, float('inf'), None
        # Loops through all 50 feature indices except the label index (index 51)
        for feature_index in range(dataset.shape[1] - 1):
            for sample in dataset:
                groups = self._split(feature_index, sample[feature_index], dataset)
                gini = self._gini(groups, class_values)
                if gini < best_score:
                    best_index, best_threshold, best_score, best_groups = feature_index, sample[feature_index], gini, groups
        return {'index': best_index, 'threshold': best_threshold, 'groups': best_groups}

    def _split(self, index, threshold, dataset):
        # Split dataset based on an index and threshold
        left = dataset[dataset[:, index] < threshold]
        right = dataset[dataset[:, index] >= threshold]
        return left, right

    def _gini(self, groups, classes):
        # Calculate the Gini index for a split
        n_instances = float(sum(len(group) for group in groups))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = sum((np.sum(group[:, -1] == class_val) / size) ** 2 for class_val in classes)
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def _to_terminal(self, group):
        # Create a terminal node by selecting the most common class
        outcomes = group[:, -1]
        return np.bincount(outcomes.astype(int)).argmax()

    def predict(self, features_dataset):
        # Predict class labels for each sample in features dataset
        return np.array([self._predict(sample, self.tree) for sample in features_dataset])

    def _predict(self, sample, node):
        # Recursively predict the class based on the tree structure
        if isinstance(node, dict):
            if sample[node['index']] < node['threshold']:
                return self._predict(sample, node['left'])
            else:
                return self._predict(sample, node['right'])
        else:
            return node