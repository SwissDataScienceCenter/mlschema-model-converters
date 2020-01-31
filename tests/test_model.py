import pytest
from mlsconverters.sklearn import to_mls
import json
import itertools

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
    PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, \
    SGDClassifier, LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, ElasticNetCV, \
    Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, ARDRegression, BayesianRidge, HuberRegressor, RANSACRegressor, \
    TheilSenRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

@pytest.mark.parametrize("sklearn_model", [
    LogisticRegression(random_state=0),
    LogisticRegressionCV(cv=5, random_state=0),
    PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3),
    Perceptron(tol=1e-3, random_state=0),
    RidgeClassifier(),
    RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]),
    SGDClassifier(max_iter=1000, tol=1e-3),
    LinearRegression(),
    Ridge(),
    RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]),
    SGDRegressor(max_iter=1000, tol=1e-3),
    ElasticNet(random_state=0),
    ElasticNetCV(cv=5, random_state=0),
    Lars(n_nonzero_coefs=1),
    LarsCV(cv=5),
    Lasso(alpha=0.1),
    LassoCV(cv=5, random_state=0),
    LassoLars(alpha=0.01),
    LassoLarsCV(cv=5),
    LassoLarsIC(criterion='bic'),
    OrthogonalMatchingPursuit(),
    OrthogonalMatchingPursuitCV(cv=5),
    ARDRegression(),
    BayesianRidge(),
    HuberRegressor(),
    RANSACRegressor(random_state=0),
    TheilSenRegressor(random_state=0),
    PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
])
def test_to_mls(sklearn_model):
    sklearn_model.fit(
        [[i+j, i+j] for i, j in itertools.product(range(5), range(3))],
        list(itertools.chain.from_iterable(itertools.repeat([0, 1, 2], 5)))
    )
    json.dumps(to_mls(sklearn_model))
