from tune_hyperparameters import optimize
from sklearn.linear_model import LogisticRegression


def weights_U_shaped(A, labels_ordered, inverse=False):
    assert n_weights%2 == 2
    x_vertice = n_weights/2; B = x_vertice*-2*A;
    return {label : A*x**2 + B*x + 1 for x, label in enumerate(labels_ordered)}

def priors_U_shaped(**kwargs):
    weights = weights_U_shaped(**kwargs)
    sum_weights = np.sum(weights.values)
    return {weights[w]/sum_weights  for w in weights}
