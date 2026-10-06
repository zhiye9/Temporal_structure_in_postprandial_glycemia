"""
Functions for the classification pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def classifiers():
    """
    Define four classifiers with tuned hyperparameters.
    """
    c_grid = [0.01, 0.1, 1, 10, 100]
    l1_logistic = LogisticRegression(C=10, penalty="l1", solver="saga", max_iter=10000, random_state=0)
    l2_logistic = LogisticRegression(C=10, penalty="l2", solver="saga", max_iter=10000, random_state=0)
    rbf_svc = SVC(kernel="rbf", C=10, probability=True, random_state=0)
    random_forest = RandomForestClassifier(random_state=23234)
    return {
        "L1_logistic": (l1_logistic, "C", c_grid),
        "L2_Logistic": (l2_logistic, "C", c_grid),
        "RBF_SVC": (rbf_svc, "C", c_grid),
        "RFR": (random_forest, "n_estimators", [20, 50, 100, 150, 200]),
    }


def reduce_features(x_fit, x_apply, pca_components):
    """
    Apply PCA under cross-validation. If pca_components is None, no PCA is applied.
    """
    if pca_components is None:
        return x_fit, x_apply
    pca = PCA(n_components=pca_components)
    return pca.fit_transform(x_fit), pca.transform(x_apply)


def inner_cv_auroc(estimator, features, labels, pos_label, splits, pca_components):
    """
    Inner cross-validation loop for candidate hyperparameter and evaluate its performance on validation set.

    Returns the AUROC scores of the inner folds.
    """
    probabilities = np.zeros(len(labels))
    for fit, validate in splits:
        x_fit, x_validate = reduce_features(features[fit], features[validate], pca_components)
        model = estimator.fit(x_fit, labels[fit])
        pos_column = list(model.classes_).index(pos_label)
        probabilities[validate] = model.predict_proba(x_validate)[:, pos_column]
    return roc_auc_score(labels == pos_label, probabilities)


def nested_cv(features, labels, pos_label, classifier_names, n_repeats, pca_components=None,
              outer_folds=5, inner_folds=4):
    """
    Run repeated stratified nested cross-validation。Both outer and inner loops are stratified k-fold splits. The inner loop is used to select the best hyperparameter for each classifier, and the outer loop is used to evaluate the performance of the selected model on test data.

    features: array (subjects x features).
    labels: metabolic subphenotype.
    pos_label: label of the positive class.
    classifier_names: list of classifiers names from classifiers().
    n_repeats: number of repetitions of the nested cross-validation.
    pca_components: use PCA under the nested cross-validation, binary variable.
    outer_folds, inner_folds: number of folds of the outer and the inner CV.

    Returns a DataFrame with classifier settings and classification results.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    models = classifiers()

    rows = []
    for repeat in range(n_repeats):
        outer = StratifiedKFold(outer_folds, shuffle=True, random_state=repeat)
        for fold, (train, test) in enumerate(outer.split(features, labels)):
            x_train, y_train = features[train], labels[train]
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=repeat * outer_folds + fold)
            inner_splits = list(inner.split(x_train, y_train))
            x_fit, x_test = reduce_features(x_train, features[test], pca_components)

            for name in classifier_names:
                estimator, parameter, grid = models[name]
                scores = []
                for value in grid:
                    candidate = estimator.set_params(**{parameter: value})
                    scores.append(inner_cv_auroc(candidate, x_train, y_train, pos_label, inner_splits, pca_components))
                best_model = grid[int(np.argmax(scores))]

                model = estimator.set_params(**{parameter: best_model}).fit(x_fit, y_train)
                auroc = round(roc_auc_score(labels[test], model.predict_proba(x_test)[:, 1]), 4)
                rows.append((repeat, fold, name, f"{parameter}={best_model}", auroc))
    return pd.DataFrame(rows, columns=["repeat", "fold", "classifier", "hyperparameter", "auroc"])
