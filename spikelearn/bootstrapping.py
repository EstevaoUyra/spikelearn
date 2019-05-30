import numpy as np
import pandas as pd
from functools import partial

def bootstrapping_analysis(data, analysis, shuffler, n_bootstrap=1000, statistics=None, agg=None):
    """
    Generic bootstrapping analysis tool
    """
    
    if statistics is None:
        statistics = lambda x: x
    
    results = []
    for iteration in range(n_bootstrap):
        results.append(statistics(analysis(shuffler(data))))

    if agg is not None:
        return agg(results)
    return results
    

def df_shuffler(df, col='time', each='trial'):
    """
    the column 'col' will have its indexes shuffled for each 'each'
    """
    assert col in df.columns and each in df.columns
    
    origin = df[col].unique()
    mapper = {e: dict(zip(origin, np.random.permutation(origin))) for e in df[each].unique()}
        
    df = df.copy()
    df[col] = df.apply(lambda row: mapper[row[each]][row[col]], axis=1)
    return df

from functools import partial
from itertools import product

def funcseq(funcs, **kwargs):
    x = funcs[0](**kwargs)
    for func in funcs[1:]:
        x = func(x)

    return x

    
class DAG_analysis():
    """
    Makes a direct acyclic graph analysis
    """
    def __init__(self, funcs=None):
        """
        funcs: list of functions or callable, optional
        """
        self.funcs = []
        self.names = []
        
        if funcs is None:
            pass
        elif type(funcs) is list:
            [self.add_step(func) for func in funcs]
        elif callable(funcs):
            self.add_step(funcs)
        else:
            raise ValueError("Funcs must be a callable or a list")
        self.pipeline = None
        
        self.compile()
        
    def __call__(self, **kwargs):
        return {key: funcseq(funcs, **kwargs) for key, funcs in self.pipeline.items()}
    
    def add_step(self, step, names=None, **kwargs):
        """
        Adds a pipeline step, optionally adding names
        Tuples are interpreted as branches
        """
        if callable(step):
            step = (partial(step, **kwargs),)
        if type(step) != tuple:
            raise ValueError("Step must be a callable or tuple of callables to be added to the pipeline")
            
        if names is None:
            names = tuple(str(f) for f in step)
        else:
            if type(names) is str:
                assert len(step) == 1
            else:
                assert len(names) == len(step)
            
        self.funcs.append(step)
        self.names.append(names)
        self.compile()
        
    def add_step_branching_by_parameter(self, func, param_name, param_values, branch_names=None, **kwargs):
        """

        """
        funcs = []
        for value in param_values:
            val = {param_name:value}
            funcs.append(partial(func, **val, **kwargs))
        
        if branch_names is not None:
            self.add_step(tuple(funcs), branch_names)
        else:
            self.add_step(tuple(funcs))

    def __repr__(self):
        return '\n'.join([str(step) for step in self.pipeline])
            
    def compile(self):
        self.pipeline = dict(zip(list(product(*self.names)),
                                 list(product(*self.funcs))))