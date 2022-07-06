from dataclasses import dataclass


@dataclass
class LeafConstraints:
    features: list[int]
    thresholds: list[float]
    directions: list[int]
    leaf_values: list[float]


class TreeConstraints:
    def __init__(self, tree: dict) -> None:
        self.constraints = self._find_paths(tree)

    @staticmethod
    def _find_paths(tree: dict) -> list[LeafConstraints]:
        """
        :param tree
        """
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
