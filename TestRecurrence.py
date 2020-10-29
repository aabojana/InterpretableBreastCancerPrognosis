# basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro

# pre-processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from ReliefF import ReliefF
from sklearn.feature_selection import RFE
from MRMR import MRMR

# models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# post-processing
from ModelsEvaluation import ModelsEvaluation
import shap

# 10 x 10-fold CV
N_outer = 10
N_inner = 10

me = ModelsEvaluation(N_outer, N_inner)


def loadFile(filename, categorical_indexes, binarized, norm):
    """
    Load dataset and transform data to be python readable.
    ----------
    :param filename : string
        The path to the file in csv format.
    :param categorical_indexes : list
        List of the indices of the categorical features.
    :param binarized: boolean
        Flag to indicate whether binarization should be performed.
    :param norm : boolean
        Flag to indicate whether standardization should be performed.
    Returns
    -------
    :return X : array-like, shape (number of samples, number of features)
        Dataset.
    :return y : array-like, shape (number of samples, number of features)
        Labels.
    """

    my_data = np.genfromtxt(filename, delimiter=',', dtype=str)
    data = np.delete(my_data, 0, 0)  # delete first raw with textual description of variables

    # define training and target data
    y = data[:, -1]
    X = np.delete(data, -1, 1)

    categorical_feature_names = {}
    new_cat_ind = 0
    X_binarized = X
    for cat_ind in categorical_indexes:
        le = LabelEncoder().fit(X[:, cat_ind])
        X[:, cat_ind] = le.transform(X[:, cat_ind])
        categorical_feature_names[str(cat_ind)] = le.classes_

        X = X.astype(float)
        y = y.astype(float)

        if (binarized):
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(le.transform(categorical_feature_names[str(cat_ind)]).reshape(-1, 1))
            encoded_categorical = encoder.transform(X[:, cat_ind].reshape(-1, 1)).toarray()
            new_cat_ind += cat_ind
            X_binarized = np.concatenate((X_binarized[:, 0:new_cat_ind], encoded_categorical,
                                          X_binarized[:, new_cat_ind + 1:]),
                                         axis=1)

    if norm:
        X_binarized_norm = StandardScaler().fit_transform(X_binarized)
        return X_binarized_norm, y

    else:
        return X_binarized, y


def risk_assessment_plot(X_binarized_norm, y, ROC_models):
    """
    Function that generates risk assessment plot. Takes as the input matrix of metrics.
    :return: plot
    """

    # for recurrence problem
    lr_probabilities, rf_probabilities, mlp_probabilities, xgb_probabilities = me.best_models_ROC_curves(
        X_binarized_norm,
        y, ROC_models,
        False)

    # for recurrence problem
    output_list1 = me.risk_assessment_data(lr_probabilities, rf_probabilities, y)
    output_list2 = me.risk_assessment_data(mlp_probabilities, xgb_probabilities, y)

    import rpy2.robjects.numpy2ri as rpyn
    rpyn.activate()

    [xrf_sens, yrf_sens, xrf_spec, yrf_spec, xlr_sens, ylr_sens, xlr_spec, ylr_spec] = rpyn.ri2py(output_list1)
    [xmlp_sens, ymlp_sens, xmlp_spec, ymlp_spec, xxgb_sens, yxgb_sens, xxgb_spec, yxgb_spec] = rpyn.ri2py(
        output_list2)

    plt.figure()
    plt.rcParams["figure.figsize"] = (10, 6)

    # for recurrence problem
    plt.plot(xlr_sens, ylr_sens, color='b',
             label='Events LR',
             lw=2, alpha=.8)
    plt.plot(xlr_spec, ylr_spec, color='b', linestyle='--',
             label='Non-events LR',
             lw=2, alpha=.8)
    plt.plot(xrf_sens, yrf_sens, color='g',
             label='Events RF',
             lw=2, alpha=.8)
    plt.plot(xrf_spec, yrf_spec, color='g', linestyle='--',
             label=r'Non-events RF',
             lw=2, alpha=.8)
    plt.plot(xmlp_sens, ymlp_sens, color='k',
             label=r'Events MLP',
             lw=2, alpha=.8)
    plt.plot(xmlp_spec, ymlp_spec, color='k', linestyle='--',
             label=r'Non-events MLP',
             lw=2, alpha=.8)
    plt.plot(xxgb_sens, yxgb_sens, color='magenta',
             label=r'Events XGBoost',
             lw=2, alpha=.8)
    plt.plot(xxgb_spec, yxgb_spec, color='magenta', linestyle='--',
             label=r'Non-events XGBoost',
             lw=2, alpha=.8)

    plt.xlabel('Calculated risk')
    plt.ylabel('Sensitivity, 1-Specificity')
    plt.legend(loc="best")
    plt.show()


def nri_idi(X_binarized_norm, y, ROC_models, file_results):
    """
    :param X_binarized_norm: array-like, shape (n_samples, n_features)
        Training data samples.
    :param y: array-like, shape (n_samples, )
        True labels.
    :param ROC_models: dictionary
        Selected classifiers with details for grid-search and feature selection.

    :return:
        Save calculated values in csv file.
    """

    # for recurrence problem
    lr_probabilities, rf_probabilities, mlp_probabilities, xgb_probabilities = me.best_models_ROC_curves(
        X_binarized_norm, y,
        ROC_models, False)

    # t parameter for NRI calculation is set  to the ratio of minority class (prevalence of minority class)
    output_list1 = me.reclassification_metrics(rf_probabilities, mlp_probabilities, y, t=0.245)
    output_list2 = me.reclassification_metrics(lr_probabilities, mlp_probabilities, y, t=0.245)
    output_list3 = me.reclassification_metrics(mlp_probabilities, xgb_probabilities, y, t=0.245)

    rf_mlp_filename = file_results + 'RF_MLP.csv'
    lr_mlp_filename = file_results + 'LR_MLP.csv'
    mlp_xgb_filename = file_results + 'MLP_XGB.csv'

    write_file = ro.r['write.csv']

    # for survival problem
    write_file(output_list1, rf_mlp_filename)
    write_file(output_list2, lr_mlp_filename)
    write_file(output_list3, mlp_xgb_filename)

    return




def bar_color(data, color1, color2):
    return np.where(data > 0, color1, color2).T


def explainMLP(filename, filename_output, csvfile_selectedData, csvfile_selctedFeatureIndices):
    """
    Function that save SHAP explainer data for defined classifier using the joblib function.

    :param filename: String
        Path to the file of input data.
    :param filename_output: String
        Path to the file to save explainer data.
    :param csvfile_selectedData: String
        Path to the file to save selected data.
    :param csvfile_selctedFeatureIndices:
        Path to the file to save indices of selected features.
    :return:
    """
    X_binarized_norm, y = loadFile(filename, [13], binarized=True, norm=True)
    clf = MLPClassifier(max_iter=2000, activation='logistic', learning_rate_init=0.1, solver='sgd',
                        hidden_layer_sizes=(20,), random_state=88)

    fs = MRMR(n_features=40)
    X_selected = fs.fit_transform(X_binarized_norm, y)
    pd.DataFrame(X_selected).to_csv(csvfile_selectedData, header=True)

    # SMOTE on training set
    sm = SMOTE(random_state=88)
    X_sm, y_sm = sm.fit_sample(X_selected, y)

    clf.fit(X_sm, y_sm)
    indices = fs.selected_features
    pd.DataFrame(indices).to_csv(csvfile_selctedFeatureIndices, header=True)

    explainer = shap.KernelExplainer(clf.predict_proba, X_selected)

    from joblib import dump
    dump(explainer, filename_output)


# MAIN PROGRAM
# ------------------------------------------------------------------------------

filename = 'Dataset/relapse.csv'
X_binarized_norm, y = loadFile(filename, [13], binarized=True, norm=True)

# List of models dictionary to evaluate
models = [{'name': 'lr', 'label': 'Logistic Regression',
           'classifier': LogisticRegression(max_iter=2000, random_state=88),
           'grid': {"classifier__C": np.logspace(-1, 1, 3)}},

          {'name': 'dt', 'label': 'Decision Tree',
           'classifier': DecisionTreeClassifier(random_state=88), 'grid': {}},

          {'name': 'rf', 'label': 'Random Forest',
           'classifier': RandomForestClassifier(random_state=88),
           'grid': {'classifier__n_estimators': [100, 200]}},

          {'name': 'svm_rbf', 'label': 'SVC (RBF)',
           'classifier': SVC(random_state=88, probability=True),
           'grid': {'classifier__C': [0.1, 1, 10, 100], 'classifier__gamma': [1, 0.1, 0.01]}},

          {'name': 'mlp', 'label': 'Multi-layer Perceptron',
           'classifier': MLPClassifier(max_iter=2000, activation='logistic', learning_rate_init=0.1, solver='sgd',
                                       random_state=88),
           'grid': {'classifier__hidden_layer_sizes': [(10,), (15,), (20,)], 'classifier__momentum': [0.1, 0.5, 0.8]}},

          {'name': 'XGBoost_Tree', 'label': 'XGBoost classifier',
           'classifier': XGBClassifier(random_state=88),
           'grid': {'classifier__n_estimators': [10, 50, 100]}}]

# list of selected features
number_of_features_to_select = [20, 25, 30, 35, 40, 45, 50]

# Evaluation of models and FS methods
# MRMR algorithm for FS
# df_aucs, df_res, df_stds = me.mrmr_feature_selection(X_binarized_norm, y, models, number_of_features_to_select)
# RFE algorithm for FS
# df_aucs, df_res, df_stds = me.rfe_feature_selection(X_binarized_norm, y, models, number_of_features_to_select)
df_aucs, df_res, df_stds = me.relieff_feature_selection(X_binarized_norm, y, models, number_of_features_to_select)

# Saving results to csv files
pathfile_results = 'Set path to the folder with results'

df_aucs.to_csv(pathfile_results + 'AUCS.csv', header=True)
df_res.to_csv(pathfile_results + 'MeanResults.csv', header=True)
df_stds.to_csv(pathfile_results + 'StdResults.csv', header=True)

# The form of the best selected models
ROC_models = [{"name": "LR_RFE_20",
               "classifier": LogisticRegression(max_iter=2000, random_state=88),
               "grid": {"classifier__C": np.logspace(-2, 2, 3)},
               "fs_method": RFE(estimator=LogisticRegression(max_iter=2000, random_state=88),
                                n_features_to_select=20,
                                step=1),
               "color_line": "g",
               "color_shadow": "yellow"},

              {"name": 'RF_ReliefF_35',
               "classifier": RandomForestClassifier(random_state=88),
               "grid": {'classifier__n_estimators': [100, 200]},
               "fs_method": ReliefF(n_neighbors=10, n_features_to_keep=35),
               "color_line": "b",
               "color_shadow": "lightblue"},

              {"name": "MLP_mRMR_40",
               "classifier": MLPClassifier(max_iter=2000, activation="logistic", learning_rate_init=0.1,
                                           hidden_layer_sizes=20, momentum=0.2, solver="sgd", random_state=88),
               "grid": {},
               "fs_method": MRMR(n_features=40),
               "color_line": 'magenta',
               "color_shadow": 'thistle'},

              {'name': 'XGBoost_RFE_30',
               'classifier': XGBClassifier(random_state=88),
               'grid': {'classifier__n_estimators': [10, 50, 100]},
               'fs_method': RFE(estimator=XGBClassifier(random_state=88),
                                n_features_to_select=30,
                                step=1),
               'color_line': 'magenta',
               'color_shadow': 'thistle'}]

# Plotting ROC curves
me.best_models_ROC_curves(X_binarized_norm, y, ROC_models, show_plot=True)

# Plotting RAP
risk_assessment_plot()

# Calculation of NRI, IDI, and relevant event/nonevent statistics
nri_idi(X_binarized_norm, y, ROC_models)

# Calculation of reliability estimations
reliability_results = me.calculate_reliability(X_binarized_norm, y, ROC_models, pathfile_results)
reliability_results.to_csv(pathfile_results + 'reliability_correlation.csv', header=True)

# Save explainer data for MLP classifier to be used in Jupyter notebook
filename_pickle = pathfile_results + 'mlp_explainer.joblib'
explainMLP(filename, filename_pickle)

print('Completed')
