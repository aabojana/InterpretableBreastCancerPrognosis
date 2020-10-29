# basic libraries
import numpy as np
import pandas as pd
import csv
from numpy import interp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import copy
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

# pre-processing
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
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

# evaluation
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, \
    cohen_kappa_score


class ModelsEvaluation:
    """
    Representes general methods for evaluation of classifiers:
        - calculation of averaged classification statics within repeated CV procedure for specific feature selection
        algorithm and several classification algorithms simultaneously with hyperparameters optimization and different
        number of selected features;
        - plotting the ROC curves of best classifiers;
        - calculates Risk Assessment Data which can be plotted for representation of event/nonevents curves
        (Risk Assessment Plot) and explain models in more comprehensive manner;
        - calculates specific reclassification metrics with confidence intervals (NRI, IDI, etc.)
        - estimation of reliability of classifiers.

        Parameters
        ----------
        n_outer : int
            Number of CV repeats.
        n_inner : int
            Number of CV repeats.
        mean_fprs: np.array
            FPR values for plotting ROC.
    """

    def __init__(self, n_outer, n_inner):
        self.N_outer = n_outer
        self.N_inner = n_inner
        self.mean_fprs = np.linspace(0, 1, 100)

    def get_specificity(self, y_true, y_pred):
        """
        Calculates the Specificity parameter.
        ----------
        :param y_true : array-like, shape (number of test samples, )
            Labels.
        :param y_pred : array-like, shape (number of test samples, )
            Predicted labels.
        Returns
        -------
        :return specif : float
            Calculated specificity parameter.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (tn + fp) == 0:
            specif = 0
        else:
            specif = tn / (tn + fp)
        return specif

    def rfe_feature_selection(self, input, output, dict_of_models, list_number_of_features_to_select):
        """
        Performs models evaluation within the No_outer times repeated No_inner-fold cross-validation procedure for different
        number of features selected by RFE algorithm with nested 10-times cross-validation for model hyperparameters' tuning
        ----------
        :param input : array-like, shape (n_samples, n_features)
            The training input samples.
        :param output : array-like, shape (n_samples, 1)
            The target values.
        :param dict_of_models: dictionary
            Models with details for grid-search.
        :param list_number_of_features_to_select - list
            Number of features to select.

        :return df_aucs : DataFrame object, shape (No_outer x No_inner, number of models x length of list_number_of_features)
            AUC values for every step of No_outer x No_inner-times CV are provided.
        :return df_res : DataFrame object, shape ([number of models x length of list_number_of_features_to_select], 9)
            For every model and every No. of selected features best classifier's parameters and averaged classification
            metrics are provided : Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC.
        :return df_stds : DataFrame object, shape ([number of models x length of list_number_of_features_to_select], 8)
            For every model and every No. of selected features standard deviations of classification metrics
            are provided.
        """

        df_res = pd.DataFrame(columns=['FS Method', 'Classifier', 'Selected features', 'Best parameters', 'Accuracy',
                                       'Sensitivity', 'Specificity', 'Precision', 'F1-score', 'ROC_AUC'])

        df_stds = pd.DataFrame(columns=['FS Method', 'Classifier', 'Selected features', 'Acc_std', 'Sens_std',
                                        'Spec_std', 'Prec_std', 'F1_std', 'ROC_AUC_std'])

        df_aucs = pd.DataFrame()

        for m in dict_of_models:
            for k in list_number_of_features_to_select:
                accuracy = []
                aucs = []
                sensitivity = []
                specificity = []
                precision = []
                f1score = []
                cohen_kappa = []
                tprs = []
                params = []
                X, y = input, output
                skf = RepeatedStratifiedKFold(n_splits=self.N_inner, n_repeats=self.N_outer, random_state=88)

                clf = m['classifier']

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    best_params = []

                    rfe = RFE(estimator=clf, n_features_to_select=k, step=1)

                    rfe_smote_clf = Pipeline([('oversampling', SMOTE(random_state=88)),
                                              ('feature_selection', rfe),
                                              ('classifier', clf)])

                    param_grid = m['grid']

                    gridsearch_cv = GridSearchCV(rfe_smote_clf,
                                                 param_grid,
                                                 cv=10,
                                                 scoring='roc_auc')

                    gridsearch_cv.fit(X_train, y_train)
                    best_params.append(gridsearch_cv.best_params_)

                    # predicted class
                    y_predict = gridsearch_cv.predict(X_test)

                    # predicted probabilities
                    probas_ = gridsearch_cv.predict_proba(X_test)

                    # accuracy
                    acc = accuracy_score(y_predict, y_test)
                    accuracy.append(acc)

                    # sensitivity = recall
                    sens = recall_score(y_test, y_predict)
                    sensitivity.append(sens)

                    # specificity
                    spec = self.get_specificity(y_test, y_predict)
                    specificity.append(spec)

                    # precision
                    prec = precision_score(y_test, y_predict)
                    precision.append(prec)

                    # f1-score
                    f1 = f1_score(y_test, y_predict)
                    f1score.append(f1)

                    # cohen-kappa-score
                    kappa = cohen_kappa_score(y_test, y_predict)
                    cohen_kappa.append(kappa)

                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1])
                    tprs.append(interp(self.mean_fprs, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)

                    # best parameters
                    params.append(best_params)

                df_aucs[m['name'] + str(k)] = aucs

                df_stds = df_stds.append({'Classifier': m['name'],
                                          'Selected features': k,
                                          'Acc_std': np.std(accuracy),
                                          'Sens_std': np.std(sensitivity),
                                          'Spec_std': np.std(specificity),
                                          'Prec_std': np.std(precision),
                                          'F1_std': np.std(f1score),
                                          'ROC_AUC_std': np.std(aucs)}, ignore_index=True)

                df_res = df_res.append({'Classifier': m['name'],
                                        'Selected features': k,
                                        'Best parameters': params,
                                        'Accuracy': np.mean(accuracy),
                                        'Sensitivity': np.mean(sensitivity),
                                        'Specificity': np.mean(specificity),
                                        'Precision': np.mean(precision),
                                        'F1-score': np.mean(f1score),
                                        'ROC_AUC': np.mean(aucs)}, ignore_index=True)

        return df_aucs, df_res, df_stds

    def relieff_feature_selection(self, input, output, dict_of_models, list_number_of_features_to_select):
        """
        Performs models evaluation within the No_outer times repeated No_inner-fold cross-validation procedure
        for different number of features selected by ReliefF algorithm with nested 10-times cross-validation for
        model hyperparameters' tuning
        ----------
        :param input : array-like, shape (n_samples, n_features)
            The training input samples.
        :param output : array-like, shape (n_samples, 1)
            The target values.
        :param dict_of_models: dictionary
            Models with details for grid-search.
        :param list_number_of_features_to_select - list
            Number of features to select.

        :return df_aucs : DataFrame object, shape (No_outer x No_inner, number of models x length of list_number_of_features)
            AUC values for every step of No_outer x No_inner-times CV are provided.
        :return df_res : DataFrame object, shape ([number of models x length of list_number_of_features_to_select], 9)
            For every model and every No. of selected features best classifier's parameters and averaged classification
            metrics are provided : Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC.
        :return df_stds : DataFrame object, shape ([number of models x length of list_number_of_features_to_select], 8)
            For every model and every No. of selected features standard deviations of classification metrics
            are provided.
        """

        df_res = pd.DataFrame(columns=['Classifier', 'Selected features', 'Best parameters', 'Accuracy', 'Sensitivity',
                                       'Specificity', 'Precision', 'F1-score', 'ROC_AUC'])

        df_stds = pd.DataFrame(columns=['Classifier', 'Selected features', 'Acc_std', 'Sens_std',
                                        'Spec_std', 'Prec_std', 'F1_std', 'ROC_AUC_std'])

        df_aucs = pd.DataFrame()

        for m in dict_of_models:
            for k in list_number_of_features_to_select:
                accuracy = []
                aucs = []
                sensitivity = []
                specificity = []
                precision = []
                f1score = []
                tprs = []
                params = []
                X, y = input, output
                skf = RepeatedStratifiedKFold(n_splits=self.N_inner, n_repeats=self.N_outer, random_state=88)

                clf = m['classifier']

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    best_params = []

                    relieff = ReliefF(n_neighbors=10, n_features_to_keep=k)

                    reliefF_smote_clf = Pipeline([('oversampling', SMOTE(random_state=88)),
                                                  ('feature_selection', relieff),
                                                  ('classifier', clf)])

                    param_grid = m['grid']

                    gridsearch_cv = GridSearchCV(reliefF_smote_clf,
                                                 param_grid,
                                                 cv=10,
                                                 scoring='roc_auc')

                    gridsearch_cv.fit(X_train, y_train)
                    best_params.append(gridsearch_cv.best_params_)

                    # predicted class
                    y_predict = gridsearch_cv.predict(X_test)

                    # predicted probabilities
                    probas_ = gridsearch_cv.predict_proba(X_test)

                    # accuracy
                    acc = accuracy_score(y_predict, y_test)
                    accuracy.append(acc)

                    # sensitivity = recall
                    sens = recall_score(y_test, y_predict)
                    sensitivity.append(sens)

                    # specificity
                    spec = self.get_specificity(y_test, y_predict)
                    specificity.append(spec)

                    # precision
                    prec = precision_score(y_test, y_predict)
                    precision.append(prec)

                    # f1-score
                    f1 = f1_score(y_test, y_predict)
                    f1score.append(f1)

                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1])
                    tprs.append(interp(self.mean_fprs, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)

                    # best parameters
                    params.append(best_params)

                df_aucs[m['name'] + str(k)] = aucs

                df_stds = df_stds.append({'Classifier': m['name'],
                                          'Selected features': k,
                                          'Acc_std': np.std(accuracy),
                                          'Sens_std': np.std(sensitivity),
                                          'Spec_std': np.std(specificity),
                                          'Prec_std': np.std(precision),
                                          'F1_std': np.std(f1score),
                                          'ROC_AUC_std': np.std(aucs)}, ignore_index=True)

                df_res = df_res.append({'Classifier': m['name'],
                                        'Selected features': k,
                                        'Best parameters': params,
                                        'Accuracy': np.mean(accuracy),
                                        'Sensitivity': np.mean(sensitivity),
                                        'Specificity': np.mean(specificity),
                                        'Precision': np.mean(precision),
                                        'F1-score': np.mean(f1score),
                                        'ROC_AUC': np.mean(aucs)}, ignore_index=True)

        return df_aucs, df_res, df_stds

    def mrmr_feature_selection(self, input, output, dict_of_models, list_number_of_features_to_select):
        """
        Performs models evaluation within the No_outer times repeated No_inner-fold cross-validation procedure
        for different number of features selected by mRMR algorithm with nested 10-times cross-validation for
        model hyperparameters' tuning
        ----------
        :param input : array-like, shape (n_samples, n_features)
            The training input samples.
        :param output : array-like, shape (n_samples, 1)
            The target values.
        :param dict_of_models: dictionary
            Models with details for grid-search.
        :param list_number_of_features_to_select - list
            Number of features to select.

        :return df_aucs : DataFrame object, shape (No_outer x No_inner, number of models x length of list_number_of_features)
            AUC values for every step of No_outer x No_inner-times CV are provided.
        :return df_res : DataFrame object, shape ([number of models x length of list_number_of_features_to_select], 9)
            For every model and every No. of selected features best classifier's parameters and averaged classification
            metrics are provided : Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC.
        :return df_stds : DataFrame object, shape ([number of models x length of list_number_of_features_to_select], 8)
            For every model and every No. of selected features standard deviations of classification metrics are provided.
        """

        df_res = pd.DataFrame(columns=['Classifier', 'Selected features', 'Best parameters', 'Accuracy', 'Sensitivity',
                                       'Specificity', 'Precision', 'F1-score', 'ROC_AUC'])

        df_stds = pd.DataFrame(columns=['Classifier', 'Selected features', 'Acc_std', 'Sens_std',
                                        'Spec_std', 'Prec_std', 'F1_std', 'ROC_AUC_std'])

        df_aucs = pd.DataFrame()

        for m in dict_of_models:
            for k in list_number_of_features_to_select:
                accuracy = []
                aucs = []
                sensitivity = []
                specificity = []
                precision = []
                f1score = []
                tprs = []
                params = []
                X, y = input, output
                skf = RepeatedStratifiedKFold(n_splits=self.N_inner, n_repeats=self.N_outer, random_state=88)

                clf = m['classifier']

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    best_params = []

                    # MRMR
                    mrmr = MRMR(n_features=k)

                    mrmr_smote_clf = Pipeline([('oversampling', ADASYN(random_state=88)),
                                               ('feature_selection', mrmr),
                                               ('classifier', clf)])

                    mrmr_smote_clf.fit(X_train, y_train)
                    # best_params.append(gridsearch_cv.best_params_)

                    # predicted class
                    y_predict = mrmr_smote_clf.predict(X_test)

                    # predicted probabilities
                    probas_ = mrmr_smote_clf.predict_proba(X_test)

                    # confusion matrix
                    cm = confusion_matrix(y_test, y_predict)

                    # accuracy
                    acc = accuracy_score(y_predict, y_test)
                    accuracy.append(acc)

                    # sensitivity = recall
                    sens = recall_score(y_test, y_predict)
                    sensitivity.append(sens)

                    # specificity
                    spec = self.get_specificity(y_test, y_predict)
                    specificity.append(spec)

                    # precision
                    prec = precision_score(y_test, y_predict)
                    precision.append(prec)

                    # f1-score
                    f1 = f1_score(y_test, y_predict)
                    f1score.append(f1)

                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1])
                    tprs.append(interp(self.mean_fprs, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)

                    # best parameters
                    params.append(best_params)

                df_aucs[m['name'] + str(k)] = aucs

                df_stds = df_stds.append({'Classifier': m['name'],
                                          'Selected features': k,
                                          'Acc_std': np.std(accuracy),
                                          'Sens_std': np.std(sensitivity),
                                          'Spec_std': np.std(specificity),
                                          'Prec_std': np.std(precision),
                                          'F1_std': np.std(f1score),
                                          'ROC_AUC_std': np.std(aucs)}, ignore_index=True)

                df_res = df_res.append({'Classifier': m['name'],
                                        'Selected features': k,
                                        'Best parameters': params,
                                        'Accuracy': np.mean(accuracy),
                                        'Sensitivity': np.mean(sensitivity),
                                        'Specificity': np.mean(specificity),
                                        'Precision': np.mean(precision),
                                        'F1-score': np.mean(f1score),
                                        'ROC_AUC': np.mean(aucs)}, ignore_index=True)

        return df_aucs, df_res, df_stds

    def best_models_ROC_curves(self, input, output, models_dict, show_plot):
        """
            Plot ROC curves for the best evaluated methods or returns averaged predicted probabilities for every sample
            by all selected models
            Parameters
            ----------
            :param input : array-like, shape (n_samples, n_features)
                The training input samples.
            :param output : array-like, shape (n_samples, 1)
                The target values.
            :param models_dict : dictionary
                Models with details for grid-search
            :param show_plot - boolean
                Indicator whether plot should be rendered

            :return [selected_models]_proba_[number_of_selected_features] : array-like, shape (1, n_samples)
                Averaged predicted probabilities for every sample by every selected model
            :return plot
        """

        X, y = input, output
        mean_fpr = np.linspace(0, 1, 100)

        instances = X.shape[0]
        proba = np.zeros((len(models_dict), 10, instances))

        skf = RepeatedStratifiedKFold(n_splits=self.N_inner, n_repeats=self.N_outer, random_state=88)

        plt.figure(1)
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='brown',
                 label='Chance', alpha=.8)

        j = 0
        for m in models_dict:
            tprs = []
            aucs = []
            clf = m['classifier']
            fs = m['fs_method']
            color_line = m['color_line']
            color_shadow = m['color_shadow']

            k = 1
            i = 0
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                fs_smote_clf = Pipeline([('oversampling', SMOTE(random_state=88)),
                                         ('feature_selection', fs),
                                         ('classifier', clf)])

                param_grid = m['grid']
                # for survival problem
                if m['name'] == 'MLP_mRMR' or m['name'] == 'SVM_mRMR_50':
                    gridsearch_cv = fs_smote_clf
                else:
                    gridsearch_cv = GridSearchCV(fs_smote_clf,
                                                 param_grid,
                                                 cv=10,
                                                 scoring='roc_auc')

                gridsearch_cv.fit(X_train, y_train)

                # predicted probabilities
                probas_ = gridsearch_cv.predict_proba(X_test)

                if k > self.N_outer:
                    break
                elif i >= k * self.N_inner:
                    k += 1

                proba[j, k - 1, test_index] = probas_[:, 1]

                # Compute ROC curve
                fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1])
                tprs.append(interp(self.mean_fprs, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                i += 1

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=color_line,
                     label=r'Mean ROC %s (AUC = %0.2f $\pm$ %0.2f)' % (m['name'], mean_auc, std_auc),
                     lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color_shadow, alpha=.2,
                             label=r'$\pm$ 1 std. dev. for ' + m['name'])

            j += 1

        if show_plot:
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.show()
            return
        else:
            mean_proba = np.mean(proba, axis=1)
            return mean_proba

    def risk_assessment_data(self, clf1_probas, clf2_probas, output):
        """
        Function that generates data for risk assessment plot.
        Call R function 'raplot' from R package for Risk Assessment Plot (RAP). It calculates the event/nonevent
        classification metrics. Takes as the input predicted probabilities of two classifiers and returns a matrix of
        metrics for generating plots.
        Parameters
        ----------
        :param clf1_probas : array-like, shape (n_samples, )
            The predicted probabilities of the first classifier
        :param clf2_probas : array-like, shape (n_samples, )
            The predicted probabilities of the second classifier
        :param output : array-like, shape (n_samples, 1)
            The true labels.
        :return output_list : R matrix
            Statistical metrics for use within 'raplot'.
        """

        utils = importr('utils')  # -- Only once.
        # select a mirror for R packages
        utils.install_packages('pROC', repos='https://cloud.r-project.org')

        ro.r.source('R_RiskAssessmentPlot/R/rap.R')
        risk_plot = ro.r['raplot']

        clf1_prob_vec = ro.vectors.FloatVector(clf1_probas)
        clf2_prob_vec = ro.vectors.FloatVector(clf2_probas)
        out_int_vec = ro.vectors.IntVector(output)

        output_list = risk_plot(clf1_prob_vec, clf2_prob_vec, out_int_vec)
        return output_list

    def ind_classification_accuracy(self, predicted_probas, labels):
        """
        Function to calculate the accuracy of predictions.
        :param predicted_probas: array-like, shape (n_samples, 2)
            Predicted probabilities.
        :param labels: array-like, shape (n_samples, )
            True labels.
        :return accuracy: array-like, shape (n_samples, )
        """
        accuracy = np.zeros(len(labels))

        for ind in range(len(labels)):
            label = int(labels[ind])
            accuracy[ind] = predicted_probas[ind, label]
        return accuracy

    def reclassification_metrics(self, clf1_probas, clf2_probas, output, t):
        """
        This function calculates specific reclassification metrics with confidence intervals (NRI, IDI, etc.) desribed in
        [Pickering JW, Endre ZH: New Metrics for Assessing Diagnostic Potential of Candidate Biomarkers.
        Clin J Am Soc Nephrol 7: 1355–1364, 2012, doi:10.2215/CJN.09590911.].
        The metrics are used for comparison two classifiers based on generated probabilities.

        :param clf1_probas: array-like, shape (n_samples, )
            Probabilities of predictions for the first classifier.
        :param clf2_probas: array-like, shape (n_samples, )
            Probabilities of predictions for the second classifier.
        :param output: array-like, shape (n_samples, )
            Real labels.
        :param t: float
            The risk threshold for groups (events/non-events).

        :return: R list
             R list with reclassification metrics data.
        """

        utils = importr('utils')  # -- Only once.
        # select a mirror for R packages
        utils.install_packages('pROC', repos='https://cloud.r-project.org')
        utils.install_packages('survival', repos='https://cloud.r-project.org')
        utils.install_packages('Hmisc', repos='https://cloud.r-project.org')
        utils.install_packages('pracma', repos='https://cloud.r-project.org')
        utils.install_packages('caret', repos='https://cloud.r-project.org')

        ro.r.source('R_RiskAssessmentPlot/R/rap.R')
        risk_plot = ro.r['CI.raplot']

        clf1_prob_vec = ro.vectors.FloatVector(clf1_probas)
        clf2_prob_vec = ro.vectors.FloatVector(clf2_probas)
        out_int_vec = ro.vectors.IntVector(output)

        s1 = 1 / t
        s2 = 1 / (1 - t)
        output_list = risk_plot(clf1_prob_vec, clf2_prob_vec, out_int_vec, s1, s2, t)
        return output_list

    def calculate_reliability(self, input, output, models_dict, file_path):
        """
        Calculates reliability estimations for selected models based on test data within the single CV procedure.
        The calculated estimations are compared with prediction accuracy by using Pearson's test.

        :param input: array-like, shape (n_samples, n_features)
            Training data.
        :param output: array-like, shape (n_samples, )
            True labels
        :param models_dict: dictionary
            Selected classifiers with details for grid-search and feature selection.
        :param file_path: String
            File path for saving data

        :return: DataFrame
            Calculated correlation coefficients.
        """
        from ReliabilityEstimation import ReliabilityEstimation
        X, y = input, output

        df_res = pd.DataFrame(columns=['Classifier', 'Corr_Oref', 'p_Oref', 'Corr_DENS', 'p_DENS', 'Corr_CNK', 'p_CNK',
                                       'Corr_LCV', 'p_LCV'])

        for m in models_dict:
            rel_ref = []
            rel_dens = []
            rel_cnk = []
            rel_lcv = []
            acc_probabilities = []
            predicted_label = []
            true_label = []
            clf = m['classifier']
            fs = m['fs_method']

            df_raw_res = pd.DataFrame(
                columns=['Oref', 'DENS', 'CNK', 'LCV', 'Accuracy', 'Predicted_label', 'True_label'])

            # leave one out cross - validation
            skf = RepeatedStratifiedKFold(n_splits=self.N_inner, n_repeats=1, random_state=88)

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                fs_smote_clf = Pipeline([('oversampling', SMOTE(random_state=88, k_neighbors=3)),
                                         ('feature_selection', fs),
                                         ('classifier', clf)])

                param_grid = m['grid']

                classifier = copy.deepcopy(fs_smote_clf)

                if m['name'] == 'MLP_mRMR_50':
                    gridsearch_cv = fs_smote_clf
                else:
                    gridsearch_cv = GridSearchCV(fs_smote_clf,
                                                 param_grid,
                                                 cv=10,
                                                 scoring='roc_auc')

                gridsearch_cv.fit(X_train, y_train)

                # predicted class
                y_predict = gridsearch_cv.predict(X_test)
                predicted_label.append(y_predict)

                # predicted probabilities
                probas_ = gridsearch_cv.predict_proba(X_test)
                acc = self.ind_classification_accuracy(probas_, y_test)
                acc_probabilities.append(acc)
                true_label.append(y_test)

                rel = ReliabilityEstimation()
                ref = list(map(lambda prob: rel.o_ref(prob), np.max(probas_, axis=1)))
                dens = list(map(lambda test: rel.DENS(X_train, test), X_test))
                cnk = list(
                    map(lambda test: rel.CNK(X_train, y_train, test, gridsearch_cv.predict_proba(test.reshape(1, -1))),
                        X_test))
                lcv = list(map(lambda test: rel.LCV(X_train, y_train, test, 40, classifier), X_test))

                rel_ref.append(ref)
                rel_dens.append(dens)
                rel_cnk.append(cnk)
                rel_lcv.append(lcv)

            merged_rel_ref = np.concatenate(rel_ref).ravel()
            merged_rel_dens = np.concatenate(rel_dens).ravel()
            merged_rel_cnk = np.concatenate(rel_cnk).ravel()
            merged_rel_lcv = np.concatenate(rel_lcv).ravel()
            merged_acc = np.concatenate(acc_probabilities).ravel()
            merged_predicted_labels = np.concatenate(predicted_label).ravel()
            merged_true_labels = np.concatenate(true_label).ravel()

            df_raw_res["Oref"] = merged_rel_ref
            df_raw_res["DENS"] = merged_rel_dens
            df_raw_res["CNK"] = merged_rel_cnk
            df_raw_res["LCV"] = merged_rel_lcv
            df_raw_res["Accuracy"] = merged_acc
            df_raw_res["Predicted_label"] = merged_predicted_labels
            df_raw_res["True_label"] = merged_true_labels

            df_raw_res.to_csv(file_path + 'Reliability_data_' + m['name'] + '.csv', header=True)

            correlation_ref, p_ref = spearmanr(merged_rel_ref, merged_acc)
            correlation_dens, p_dens = spearmanr(merged_rel_dens, merged_acc)
            correlation_cnk, p_cnk = spearmanr(merged_rel_cnk, merged_acc)
            correlation_lcv, p_lcv = spearmanr(merged_rel_lcv, merged_acc)

            df_res = df_res.append({'Classifier': m['name'],
                                    'Corr_Oref': correlation_ref,
                                    'p_Oref': p_ref,
                                    'Corr_DENS': correlation_dens,
                                    'p_DENS': p_dens,
                                    'Corr_CNK': correlation_cnk,
                                    'p_CNK': p_cnk,
                                    'Corr_LCV': correlation_lcv,
                                    'p_LCV': p_lcv}, ignore_index=True)

        return df_res
