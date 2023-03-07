from dataclasses import dataclass
from pyomo import environ
import numpy as np


@dataclass
class LeafConstraints:
    features: list[int]
    thresholds: list[float]
    directions: list[int]
    leaf_values: list[float]


@dataclass
class FullTreeConstraints:
    leaf_and_features: np.array
    thresholds: np.array
    predictions: np.array


class TreeConstraints:
    def __init__(self, tree: dict, n_features: int) -> None:
        self.tree = tree
        self.n_features = n_features
        self.paths = list(self._find_paths(tree))
        self.constraints = self._generate_tree_constraints(n_features)
        
    @staticmethod
    def _find_paths(tree: dict) -> list[LeafConstraints]:
        stack = [[tree, [], [], []]]
        while stack:
            node, path, thresholds, directions = stack.pop()        
            if 'leaf_value' in node:
                yield LeafConstraints(path, thresholds, directions, [node['leaf_value']] * len(path))
            else:
                path = path + [node['split_feature']]
                thresholds = thresholds + [node['threshold']]
                stack.append([node['left_child'], path, thresholds, directions + [-1]])
                stack.append([node['right_child'], path, thresholds, directions + [1]])
        return stack
    
    
    def _generate_tree_constraints(self, n_features):
        split_idc = []
        split_directions = []
        y_pred = []
        thresholds = []
        leaf_ids = []

        for i, path in enumerate(self.paths):
            split_idc.extend(path.features)
            thresholds.extend(path.thresholds)
            split_directions.extend(path.directions)
            y_pred.extend(path.leaf_values)
            leaf_ids.extend([i] * len(path.features))

        split_directions = np.array(split_directions)
        splits = np.zeros((len(split_idc), n_features))
        splits[np.arange(len(splits)), split_idc] = 1 * split_directions
        threshold = thresholds * split_directions + 0.000001 * (split_directions == -1)

        return FullTreeConstraints(
            np.hstack([np.expand_dims(leaf_ids, 1), splits]),
            threshold,
            y_pred
        )


    def _embed_tree_constraints(self, model, tree_id: int, feature_names: list[str]) -> None:
        leafs = self.constraints.leaf_and_features[:, 0].astype(int)
        features = self.constraints.leaf_and_features[:, 1:]
        leaf_ids = np.unique(leafs)
        
        def tree_constraints_1(model, row):
            return sum(model.x[feature_names[i]] * features[row, i] for i in range(self.n_features)) <= self.constraints.thresholds[row] + 1e5 * (1 - model.l[('y', str(leafs[row]))])
        
        def tree_constraints_2(model):
            return sum(model.l[(str(tree_id), str(leaf))] for leaf in leaf_ids) == 1
        
        def tree_constraint(model):
            preds = self.constraints.predictions
            return model.y[str(tree_id)] == sum(preds[i] * model.l[('y', str(i))] for i in leaf_ids)

        model.add_component(f"tree_{tree_id}_1", environ.Constraint(range(leafs.size), rule=tree_constraints_1))
        model.add_component(f"decision_tree_{tree_id}", environ.Constraint(rule=tree_constraint))
        model.add_component(f"tree_{tree_id}_2", environ.Constraint(rule=tree_constraints_2))