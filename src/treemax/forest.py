import lightgbm
from tree import TreeConstraints
from typing import Union, Optional
import numpy as np
import pandas as pd 
from pyomo import environ


class Treemax:
    def __init__(
        self, 
        model: lightgbm.sklearn.LGBMRegressor,
        X: Union[np.array, pd.DataFrame],
        y: Union[np.array, pd.DataFrame],
        sample_weights: Optional[Union[pd.DataFrame, np.array]] = None
    ) -> None:
        
        assert isinstance(model, lightgbm.sklearn.LGBMRegressor)
        self.is_embedded = False
        self.is_optimized = False

        self.booster = model
        self.X = X
        self.trees = self.__get_trees()
        self.trees = [self.__get_tree_structure(tree) for tree in self.trees]
        self.n_trees = len(self.trees)
             
        self.optimization_model = self.__make_optimization_model()
    
        self.init_score = self.__get_init_score(y, sample_weights=sample_weights)
    
    def __make_optimization_model(self):
        model = environ.ConcreteModel()
        model.x = environ.Var(self.booster.feature_name_, domain=environ.Reals)
        model.OBJ = environ.Objective(expr=0, sense=environ.maximize)
        model.y = environ.Var(environ.Any, dense=False, domain=environ.Reals)
        model.l = environ.Var(environ.Any, dense=False, domain=environ.Binary)
        model.y_viol = environ.Var(environ.Any, dense=False, domain=environ.Binary)
        model.v = environ.Var(environ.Any, dense=False, domain=environ.NonNegativeReals)
        model.v_ind = environ.Var(environ.Any, dense=False, domain=environ.Binary)
        return model
    
    
    def __get_trees(self):
        return self.booster.booster_.dump_model()['tree_info']
    
    
    def __get_tree_structure(self, tree):
        return TreeConstraints(tree['tree_structure'], self.booster.n_features_)
    
    
    def __get_init_score(self, y, sample_weights: Optional[Union[pd.DataFrame, np.array]] = None):
        if sample_weights is None:
            sample_weights = np.ones(y.size)
        return sum(y * sample_weights) / sum(sample_weights)
    
    
    def embed_model(self, weight_objective: float = 0):        
        for tree_id, tree in enumerate(self.trees):
            tree._embed_tree_constraints(
                model=self.optimization_model, 
                tree_id=tree_id,
                feature_names=self.booster.feature_name_
            )
            
        def constraint_gbm(model):
            return model.y['y'] == self.init_score + self.booster.learning_rate * environ.quicksum(model.y[str(tree_id)] for tree_id in range(self.n_trees))
        
        self.optimization_model.add_component('GBM'+'y', environ.Constraint(rule=constraint_gbm))
    
        if weight_objective != 0:
            self.optimization_model.OBJ.set_value(
                expr=self.optimization_model.OBJ.expr + weight_objective * self.optimization_model.y['y']
            )  
        
        self.is_embedded = True
