#!/usr/bin/env python
# coding: utf-8

#------ Load necessary packages ------#
import sys
import os
import numpy as np
# import dataloader
# import pandas as pd
#from common_utils import compute_metrics
# sys.path.append("../")
# load datasets
# from aif360.datasets import CelebADataset
from aif360.datasets import imsitu_dataset

# ca# load metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
# from aif360.metrics.utils import compute_boolean_conditioning_vector

# load preprocessing algorithm
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german
# from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
# from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult

# load algorithms
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

# load other packages
import tensorflow as tf
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
import argparse
# from plot import plot
from keras.models import load_model
# from sklearn.metrics import classification_report
# from IPython.display import Markdown, display
# import matplotlib.pyplot as plt


file_name = os.path.basename(__file__)
# logging(file_name, 2)

#------ Input dataset ------#
# get the adult dataset and split into train and test
# dataset_used = "adult"  # "adult", "german", "compas", "bank"
# protected_attribute_used = 1  # 1, 2

parser = argparse.ArgumentParser()
# parser.add_argument('--dropouts', nargs='+', type=float, default=[0.1,0.1,0.1])
parser.add_argument('--dataset', type=str, default='imsitu')
parser.add_argument('--senstive_feature', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_t', type=float, default=1e-4)
parser.add_argument('--network', type=str, default='res50')
args = parser.parse_args()

dataset_used = args.dataset
protected_attribute_used = args.senstive_feature
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if dataset_used == "adult":
    dataset_orig = AdultDataset()
    # dataset_orig = load_preproc_data_adult()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        sens_attr = 'sex'
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        sens_attr = 'race'

elif dataset_used == "german":
    dataset_orig = GermanDataset()
    # dataset_orig.labels = dataset_orig.labels-1
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        sens_attr = 'sex'
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        sens_attr = 'age'

elif dataset_used == "compas":
    dataset_orig = CompasDataset()
    # dataset_orig = load_preproc_data_compas()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        sens_attr = 'sex'
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        sens_attr = 'race'

elif dataset_used == "bank":
    dataset_orig = BankDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        sens_attr = 'age'
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        sens_attr = 'age'

elif dataset_used == 'celebA':
    dataset_orig = CelebADataset(network = args.network)
    privileged_groups = [{'gender': 0}]
    unprivileged_groups = [{'gender': 1}]
    sens_attr = 'gender'
    dataset_orig.features[:,0:2048] += 1e-8 # male = 2, female = 1
    
elif dataset_used == 'imsitu':
    dataset_orig = imsitu_dataset.ImsituDataset(favorable_classes=[1])
    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]
    sens_attr = 'gender'
    
repeat = args.repeat
name = args.network

total_acc = np.zeros((11, repeat))
total_balanced_acc = np.zeros((11, repeat))
total_disimpact = np.zeros((11, repeat))
total_eqopp_diff = np.zeros((11, repeat))
total_aveodds_diff = np.zeros((11, repeat))
total_theil_idx = np.zeros((11, repeat))
stat_parity_diff = np.zeros((11, repeat))

total_tpr = np.zeros((11, repeat))
total_tpr_priv = np.zeros((11, repeat))
total_tpr_unpriv = np.zeros((11, repeat))

total_fpr = np.zeros((11, repeat))
total_fpr_priv = np.zeros((11, repeat))
total_fpr_unpriv = np.zeros((11, repeat))
total_fpr_diff = np.zeros((11, repeat))

total_acc_priv = np.zeros((11, repeat))
total_acc_unpriv = np.zeros((11, repeat))


# total_acc = np.load('save_compare_data/' + name + '_total_acc.npz')['total_acc']
# total_balanced_acc = np.load('save_compare_data/' + name + '_total_balanced_acc.npz')['total_balanced_acc']
# total_disimpact = np.load('save_compare_data/' + name + '_total_disimpact.npz')['total_disimpact']
# total_eqopp_diff = np.load('save_compare_data/' + name + '_total_eqopp_diff.npz')['total_eqopp_diff']
# total_aveodds_diff = np.load('save_compare_data/' + name + '_total_aveodds_diff.npz')['total_aveodds_diff']
# total_theil_idx = np.load('save_compare_data/' + name + '_total_theil_idx.npz')['total_theil_idx']
# stat_parity_diff = np.load('save_compare_data/' + name + '_stat_parity_diff.npz')['stat_parity_diff']
# total_tpr = np.load('save_compare_data/' + name + '_total_tpr.npz')['total_tpr']
# total_fpr_diff = np.load('save_compare_data/' + name + '_total_fpr_diff.npz')['total_fpr_diff']

# dataset_orig, _ = dataset_orig.split([0.1], shuffle=True)
# #print(len(dataset_orig))
# min_max_scaler = MaxAbsScaler()
# dataset_orig.features[:,:-1] = min_max_scaler.fit_transform(dataset_orig.features)[:,:-1]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sens_idx = (dataset_orig.feature_names).index(sens_attr)

for iter in range(repeat):
    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    
    tmp = dataset_orig_train.labels.astype(int)
    if dataset_used == "german":
        tmp = tmp - 1
    targets = tmp.reshape(-1).tolist()
    y_train = np.eye(np.unique(tmp).shape[0])[targets]

    tmp = dataset_orig_valid.labels.astype(int)
    if dataset_used == "german":
        tmp = tmp - 1
    targets = tmp.reshape(-1).tolist()
    y_val = np.eye(np.unique(tmp).shape[0])[targets]

    tmp = dataset_orig_test.labels.astype(int)
    if dataset_used == "german":
        tmp = tmp - 1
    targets = tmp.reshape(-1).tolist()
    y_test = np.eye(np.unique(tmp).shape[0])[targets]
    

    

    if iter == 0:
        # print out some labels, names, etc.
        print("#------ Training Dataset Shape ------#")
        print(dataset_orig_train.features.shape)
        print("#------ Favorable and Unfavorable Labels ------#")
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        print("#------ Protected Attribute Names ------#")
        # print(dataset_orig_train.protected_attribute_names)
        print(sens_attr)
        print("#------ Privileged and Unprivileged Protected Attribute Values ------#")
        print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
        print("#------ Dataset Feature Names ------#")
        print(dataset_orig_train.feature_names)

#     normalize

    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_valid.features = min_max_scaler.fit_transform(dataset_orig_valid.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    
        # parameter setup
    thresh_arr = np.linspace(0., 1., 11)
    
   
    
#     #------ Building models ------#
#     # 10. Fairness via Shapley
#     print("#------ 10. Fairness via Shapley ------#")
    sess = tf.Session(config=config)
    sens_idx = (dataset_orig_train.feature_names).index(sens_attr)
    tmp = dataset_orig_test.labels.astype(int)
    if dataset_used == "german":
        tmp = tmp - 1
    targets = tmp.reshape(-1).tolist()
    y_train = np.eye(np.unique(tmp).shape[0])[targets]

    outpred = fair_shapley(dataset_orig_test.features, y_train, dataset_orig_test.features, sens_idx, sess,
                           args.batch_size, args.epochs, args.gpu_id)
    dataset_pred = dataset_orig_test.copy()
    if dataset_used == "german":
        dataset_pred.labels = np.argmax(outpred, axis=1) + 1
    else:
        dataset_pred.labels = np.argmax(outpred, axis=1)
    classified_metric = ClassificationMetric(dataset_orig_test,
                                             dataset_pred,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    # print("Classification accuracy = %f" % classified_metric.accuracy())
    # TPR = classified_metric.true_positive_rate()
    # TNR = classified_metric.true_negative_rate()
    # bal_acc_test = 0.5*(TPR+TNR)
    # print("Balanced classification accuracy = %f" % bal_acc_test)
    # print("Disparate impact = %f" % metric_pred.disparate_impact())
    # print("Equal opportunity difference = %f" % classified_metric.equal_opportunity_difference())
    # print("Average odds difference = %f" % classified_metric.average_odds_difference())
    # print("Theil_index = %f" % classified_metric.theil_index())
    # print("Statistical parity difference = %f" % classified_metric.statistical_parity_difference())
    total_acc[9, iter] = classified_metric.accuracy()
    total_balanced_acc[9, iter] = 0.5 * (classified_metric.true_positive_rate() +
                                         classified_metric.true_negative_rate())
    total_disimpact[9, iter] = metric_pred.disparate_impact()
    total_eqopp_diff[9, iter] = classified_metric.equal_opportunity_difference()
    total_aveodds_diff[9, iter] = classified_metric.average_odds_difference()
    total_theil_idx[9, iter] = classified_metric.theil_index()
    stat_parity_diff[9, iter] = classified_metric.statistical_parity_difference()

    total_tpr[9, iter] = classified_metric.recall()
    total_fpr_diff[9, iter] = classified_metric.false_positive_rate_difference()
    total_fpr[9, iter] = classified_metric.false_positive_rate()

    sess.close()
    tf.reset_default_graph()

#     print("EQ", classified_metric.equal_opportunity_difference())
#     print("acc",classified_metric.accuracy())
    
    
    # 2. Adversarial Debiasing model
    # Learn parameters with debias set to True
    print("#------ 2. Adversarial debiasing model ------#")
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(dataset_orig_train)

    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                            dataset_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
    # print("Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    # TPR = classified_metric_debiasing_test.true_positive_rate()
    # TNR = classified_metric_debiasing_test.true_negative_rate()
    # bal_acc_debiasing_test = 0.5*(TPR+TNR)
    # print("Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    # print("Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
    # print("Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
    # print("Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
    # print("Theil_index = %f" % classified_metric_debiasing_test.theil_index())
    # print("Statistical parity difference = %f" % classified_metric_debiasing_test.statistical_parity_difference())
    total_acc[1, iter] = classified_metric_debiasing_test.accuracy()
    total_balanced_acc[1, iter] = 0.5 * (classified_metric_debiasing_test.true_positive_rate() +
                                         classified_metric_debiasing_test.true_negative_rate())
    total_disimpact[1, iter] = classified_metric_debiasing_test.disparate_impact()
    total_eqopp_diff[1, iter] = classified_metric_debiasing_test.equal_opportunity_difference()
    total_aveodds_diff[1, iter] = classified_metric_debiasing_test.average_odds_difference()
    total_theil_idx[1, iter] = classified_metric_debiasing_test.theil_index()
    stat_parity_diff[1, iter] = classified_metric_debiasing_test.statistical_parity_difference()
    total_tpr[1, iter] = classified_metric_debiasing_test.recall()
    total_fpr_diff[1, iter] = abs(classified_metric_debiasing_test.performance_measures(True)['FPR'] - classified_metric_debiasing_test.performance_measures(False)['FPR'])
    sess.close()
    tf.reset_default_graph()


    
    

    # 3. Calibrated Equal Odds
    print("#------ 3. Calibrated equal odds ------#")
    # sess = tf.Session()
    cost_constraint = "fnr" # "fnr", "fpr", "weighted"
    randseed = 12345679
    sess = tf.Session()
    # Logistic regression classifier and predictions for training data
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    X_valid = scale_orig.transform(dataset_orig_valid.features)
    y_train = dataset_orig_train.labels.ravel()

    tmp = y_train.astype(int)
    targets = tmp.reshape(-1).tolist()
    y_train = np.eye(np.unique(tmp).shape[0])[targets]

    base_model = fair_ablation.FAIR(X_train, y_train, sens_idx, sess,\
                     args.batch_size, args.epochs)
    
    base_model.train(X_train, y_train)

    fav_idx = np.where(base_model.classes_ == dataset_orig_train.favorable_label)[0][0]

    y_train_pred_prob = base_model.get_prediction(X_train)[:, fav_idx]
    y_valid_pred_prob = base_model.get_prediction(X_valid)[:, fav_idx]

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    bal_acc_arr = []

    for class_thresh in thresh_arr:
        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)

        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred

        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred

        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                             unprivileged_groups = unprivileged_groups,
                                             cost_constraint=cost_constraint,
                                             seed=randseed)
        # cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
        cpp.fit(dataset_orig_train, dataset_orig_train_pred)
        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)

        cm_transf_valid = ClassificationMetric(dataset_orig_valid,
                                              dataset_transf_valid_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
        bal_acc = (cm_transf_valid.true_positive_rate() )
        bal_acc_arr.append(bal_acc)

    thresh_arr_best_ind = np.where(bal_acc_arr == np.max(bal_acc_arr))[0][0]
    thresh_arr_best = np.array(thresh_arr)[thresh_arr_best_ind]

    X_test = scale_orig.transform(dataset_orig_test.features)
    
    y_test_pred_prob = base_model.get_prediction(X_test)[:, fav_idx]
#    y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    y_test_pred_prob = base_model.get_prediction(X_test)[:, fav_idx]
    #y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)
    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= thresh_arr_best] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= thresh_arr_best)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred

    dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
    cm_transf_test = ClassificationMetric(dataset_orig_test,
                                          dataset_transf_test_pred,
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)

    sess.close()
    tf.reset_default_graph()
    print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr_best)
    # print("Classification accuracy = %f" % cm_transf_test.accuracy())
    # TPR = cm_transf_test.true_positive_rate()
    # TNR = cm_transf_test.true_negative_rate()
    # bal_acc_debiasing_test = 0.5*(TPR+TNR)
    # print("Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    # print("Disparate impact = %f" % cm_transf_test.disparate_impact())
    # print("Equal opportunity difference = %f" % cm_transf_test.equal_opportunity_difference())
    # print("Average odds difference = %f" % cm_transf_test.average_odds_difference())
    # print("Theil_index = %f" % cm_transf_test.theil_index())
    # print("Statistical parity difference = %f" % cm_transf_test.statistical_parity_difference())
    total_acc[2, iter] = cm_transf_test.accuracy()
    total_balanced_acc[2, iter] = 0.5 * (cm_transf_test.true_positive_rate() +
                                         cm_transf_test.true_negative_rate())
    total_disimpact[2, iter] = cm_transf_test.disparate_impact()
    total_eqopp_diff[2, iter] = cm_transf_test.equal_opportunity_difference()
    total_aveodds_diff[2, iter] = cm_transf_test.average_odds_difference()
    total_theil_idx[2, iter] = cm_transf_test.theil_index()
    stat_parity_diff[2, iter] = cm_transf_test.statistical_parity_difference()
    total_tpr[2, iter] = cm_transf_test.recall()
    total_fpr_diff[2, iter] = abs(cm_transf_test.performance_measures(True)['FPR'] - cm_transf_test.performance_measures(False)['FPR'])

    
    
    # 4. DisparateImpactRemover
    print("#------ 4. Disparate Impact Remover ------#")
    bal_acc_arr = []

    

    for level in tqdm(thresh_arr):
        sess = tf.Session()
        di = DisparateImpactRemover(repair_level=level)
        train_repd = di.fit_transform(dataset_orig_train)
        validate_repd = di.fit_transform(dataset_orig_valid)
        # test_repd = di.fit_transform(dataset_orig_test)

        X_tr = np.delete(train_repd.features, protected_attribute_used, axis=1)
        X_valid = np.delete(validate_repd.features, protected_attribute_used, axis=1)
        # X_te = np.delete(test_repd.features, protected_attribute_used, axis=1)
        y_tr = train_repd.labels.ravel()
        X_tr = X_tr.reshape(X_tr.shape[0], -1)
        y_tr = y_tr.reshape(y_tr.shape[0], -1)

        tmp = y_tr.astype(int)
        targets = tmp.reshape(-1).tolist()
        y_train = np.eye(np.unique(tmp).shape[0])[targets]


        base_model = fair_ablation.FAIR(X_tr, y_train, sens_idx, sess,\
                 args.batch_size, args.epochs)
        base_model.train(X_tr, y_train)

        # test_repd_pred = test_repd.copy()
        # test_repd_pred.labels = lmod.predict(X_te)
        # cm = BinaryLabelDatasetMetric(test_repd_pred,
        #                               privileged_groups=privileged_groups,
        #                               unprivileged_groups=unprivileged_groups)
        # DIs.append(cm.disparate_impact())
        # print("\nLevel = %f" % level)
        # print("Disparate Impact Remover = %f" % cm.disparate_impact())
        y_valid_pred_prob = base_model.get_prediction(X_valid)
#        y_valid_pred_prob = lmod.predict_proba(X_valid)
        y_valid_pred = (y_valid_pred_prob[:, 1] > level).astype(np.double)
        validate_repd_pred = validate_repd.copy()
        validate_repd_pred.labels = y_valid_pred

        classified_metric = ClassificationMetric(validate_repd,
                                                 validate_repd_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        metric_pred = BinaryLabelDatasetMetric(validate_repd_pred,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

        bal_acc = 0.5 * (classified_metric.true_positive_rate() + classified_metric.true_negative_rate())
        bal_acc_arr.append(bal_acc)

        sess.close()
        tf.reset_default_graph()

    sess = tf.Session()

    thresh_arr_best_ind = np.where(bal_acc_arr == np.max(bal_acc_arr))[0][0]
    thresh_arr_best = np.array(thresh_arr)[thresh_arr_best_ind]

    di = DisparateImpactRemover(repair_level=thresh_arr_best)
    train_repd = di.fit_transform(dataset_orig_train)
    test_repd = di.fit_transform(dataset_orig_test)

    X_tr = np.delete(train_repd.features, protected_attribute_used, axis=1)
    X_te = np.delete(test_repd.features, protected_attribute_used, axis=1)
    y_tr = train_repd.labels.ravel()
    y_te = test_repd.labels.ravel()

    tmp = y_tr.astype(int)
    targets = tmp.reshape(-1).tolist()
    y_train = np.eye(np.unique(tmp).shape[0])[targets]
    
    tmp = y_te.astype(int)
    targets = tmp.reshape(-1).tolist()
    y_test = np.eye(np.unique(tmp).shape[0])[targets]
        
    
    base_model = fair_ablation.FAIR(X_tr, y_train, sens_idx, sess,
                                    args.batch_size, args.epochs)
    base_model.train(X_tr, y_train)

    y_te_pred_prob = base_model.get_prediction(X_te)
    y_te_pred = (y_te_pred_prob[:, 1] > thresh_arr_best).astype(np.double)
    test_repd_pred = test_repd.copy()
    test_repd_pred.labels = y_te_pred

    classified_metric = ClassificationMetric(test_repd,
                                             test_repd_pred,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    metric_pred = BinaryLabelDatasetMetric(test_repd_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr_best)
    # acc = accuracy_score(y_true=test_repd.labels, y_pred=test_repd_pred.labels)
    # print("Classification accuracy = %f" % acc)
    # TPR = classified_metric.true_positive_rate()
    # TNR = classified_metric.true_negative_rate()
    # bal_acc_test = 0.5*(TPR+TNR)
    # print("Balanced classification accuracy = %f" % bal_acc_test)
    # print("Disparate impact = %f" % metric_pred.disparate_impact())
    # print("Equal opportunity difference = %f" % classified_metric.equal_opportunity_difference())
    # print("Average odds difference = %f" % classified_metric.average_odds_difference())
    # print("Theil_index = %f" % classified_metric.theil_index())
    # print("Statistical parity difference = %f" % classified_metric.statistical_parity_difference())
    total_acc[3, iter] = classified_metric.accuracy()
    total_balanced_acc[3, iter] = 0.5 * (classified_metric.true_positive_rate() +
                                         classified_metric.true_negative_rate())
    total_disimpact[3, iter] = metric_pred.disparate_impact()
    total_eqopp_diff[3, iter] = classified_metric.equal_opportunity_difference()
    total_aveodds_diff[3, iter] = classified_metric.average_odds_difference()
    total_theil_idx[3, iter] = classified_metric.theil_index()
    stat_parity_diff[3, iter] = classified_metric.statistical_parity_difference()
    total_tpr[3, iter] = classified_metric.recall()
    total_fpr_diff[3, iter] = abs(classified_metric.performance_measures(True)['FPR'] - classified_metric.performance_measures(False)['FPR'])
    sess.close()
    tf.reset_default_graph()

    '''

    # # 5. LFR
    # print("#------ 5. LFR ------#")
    # metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
    #                                              unprivileged_groups=unprivileged_groups,
    #                                              privileged_groups=privileged_groups)
    #
    # TR = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # TR = TR.fit(dataset_orig_train)
    #
    # dataset_transf_train = TR.transform(dataset_orig_train)
    # thresholds = [0.2, 0.3, 0.35, 0.4, 0.5]
    #
    # for threshold in thresholds:
    #     # Transform training data and align features
    #     dataset_transf_train = TR.transform(dataset_orig_train, threshold=threshold)
    #     metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
    #                                                    unprivileged_groups=unprivileged_groups,
    #                                                    privileged_groups=privileged_groups)
    #     print("Classification threshold = %f" % threshold)
    #     print("Difference in mean outcomes between unprivileged and privileged groups = %f"
    #           % metric_transf_train.mean_difference())
    #
    # print("#### Individual fairness metrics")
    # print("Consistency of labels in transformed training dataset= %f" % metric_transf_train.consistency())
    # print("Consistency of labels in original training dataset= %f" % metric_orig_train.consistency())




    # # 6. Optimized Preprocessing
    # print("#------ 6. Optimized Preprocessing ------#")
    # optim_options = {
    #     "distortion_fun": get_distortion_adult,
    #     "epsilon": 0.05,
    #     "clist": [0.99, 1.99, 2.99],
    #     "dlist": [.1, 0.05, 0]
    # }
    #
    # OP = OptimPreproc(OptTools, optim_options)
    # OP = OP.fit(dataset_orig_train)
    #
    # dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
    # dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
    # dataset_transf_test = OP.transform(dataset_orig_test, transform_Y=True)
    # dataset_transf_test = dataset_orig_test.align_datasets(dataset_transf_test)
    #
    # # metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
    # #                                                unprivileged_groups=unprivileged_groups,
    # #                                                privileged_groups=privileged_groups)
    # # print("Test set: Disparate impact = %f" % metric_transf_train.disparate_impact())
    # # print("Difference in mean outcomes between unprivileged and privileged groups = %f"
    # #       % metric_transf_train.mean_difference())
    #
    # classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
    #                                                         dataset_transf_test,
    #                                                         unprivileged_groups=unprivileged_groups,
    #                                                         privileged_groups=privileged_groups)
    # print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    # TPR = classified_metric_debiasing_test.true_positive_rate()
    # TNR = classified_metric_debiasing_test.true_negative_rate()
    # bal_acc_debiasing_test = 0.5*(TPR+TNR)
    # print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    # print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
    # print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
    # print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
    # print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())
    #
    # # sess.close()
    # # tf.reset_default_graph()


    
    '''

    # 7. Reweighing
    print("#------ 7. Reweighing ------#")
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)

    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    sess = tf.Session()
    sens_idx = (dataset_orig_train.feature_names).index(sens_attr)

    tmp = y_train.astype(int)
    targets = tmp.reshape(-1).tolist()
    y_train = np.eye(np.unique(tmp).shape[0])[targets]
    
   

    base_model = fair_ablation.FAIR(X_train, y_train, sens_idx, sess,\
                     args.batch_size, args.epochs)
    
    base_model.train(X_train, y_train)

    y_train_pred = base_model.get_prediction(X_train)
    
    y_train_pred = np.argmax(y_train_pred, axis=1)
    
    pos_ind = np.where(base_model.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    
    dataset_orig_valid_pred.scores = base_model.get_prediction(X_valid)[:, pos_ind].reshape(-1, 1)
    
#    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = base_model.get_prediction(X_test)[:, pos_ind].reshape(-1, 1)

    # num_thresh = 100
    # ba_arr = np.zeros(num_thresh)
    # class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    ba_arr = np.zeros(thresh_arr.shape[0])
    for idx, class_thresh in enumerate(thresh_arr):
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                            dataset_orig_valid_pred,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

        ba_arr[idx] = 0.5 * (
                classified_metric_orig_valid.true_positive_rate() + classified_metric_orig_valid.true_negative_rate())

    idx = np.argmax(ba_arr)
    fav_inds = dataset_orig_test_pred.scores > thresh_arr[idx]
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
                                                        dataset_orig_test_pred,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)

    print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr[idx])
    # print("Classification accuracy = %f" % classified_metric_orig_test.accuracy())
    # TPR = classified_metric_orig_test.true_positive_rate()
    # TNR = classified_metric_orig_test.true_negative_rate()
    # bal_acc_test = 0.5 * (TPR + TNR)
    # print("Balanced classification accuracy = %f" % bal_acc_test)
    # print("Disparate impact = %f" % classified_metric_orig_test.disparate_impact())
    # print("Equal opportunity difference = %f" % classified_metric_orig_test.equal_opportunity_difference())
    # print("Average odds difference = %f" % classified_metric_orig_test.average_odds_difference())
    # print("Theil_index = %f" % classified_metric_orig_test.theil_index())
    # print("Statistical parity difference = %f" % classified_metric_orig_test.statistical_parity_difference())
    total_acc[6, iter] = classified_metric_orig_test.accuracy()
    total_balanced_acc[6, iter] = 0.5 * (classified_metric_orig_test.true_positive_rate() +
                                         classified_metric_orig_test.true_negative_rate())
    total_disimpact[6, iter] = classified_metric_orig_test.disparate_impact()
    total_eqopp_diff[6, iter] = classified_metric_orig_test.equal_opportunity_difference()
    total_aveodds_diff[6, iter] = classified_metric_orig_test.average_odds_difference()
    total_theil_idx[6, iter] = classified_metric_orig_test.theil_index()
    stat_parity_diff[6, iter] = classified_metric_orig_test.statistical_parity_difference()
    total_tpr[6, iter] = classified_metric_orig_test.recall()
    total_fpr_diff[6, iter] = abs(classified_metric_orig_test.performance_measures(True)['FPR'] - classified_metric_orig_test.performance_measures(False)['FPR'])
    sess.close()
    tf.reset_default_graph()
    
    
    '''
    # 8. Reject Option
    print("#------ 8. Reject Option ------#")
    # # Logistic regression classifier and predictions
    # scale_orig = StandardScaler()
    # X_train = scale_orig.fit_transform(dataset_orig_train.features)
    # y_train = dataset_orig_train.labels.ravel()
    #
    # lmod = LogisticRegression()
    # lmod.fit(X_train, y_train)
    # y_train_pred = lmod.predict(X_train)
    # pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    #
    # dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    # dataset_orig_train_pred.labels = y_train_pred
    #
    # dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    # X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    # y_valid = dataset_orig_valid_pred.labels
    # dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)
    #
    # dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    # X_test = scale_orig.transform(dataset_orig_test_pred.features)
    # y_test = dataset_orig_test_pred.labels
    # dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    #
    # ba_arr = np.zeros(thresh_arr.shape[0])
    # for idx, class_thresh in enumerate(thresh_arr):
    #     fav_inds = dataset_orig_valid_pred.scores > class_thresh
    #     dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    #     dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    #
    #     classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
    #                                                         dataset_orig_valid_pred,
    #                                                         unprivileged_groups=unprivileged_groups,
    #                                                         privileged_groups=privileged_groups)
    #
    #     ba_arr[idx] = 0.5 * (
    #             classified_metric_orig_valid.true_positive_rate() + classified_metric_orig_valid.true_negative_rate())
    #
    # best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    # best_class_thresh = thresh_arr[best_ind]

    # Metric used (should be one of allowed_metrics)
    metric_name = "Statistical parity difference"
    metric_ub = 0.05
    metric_lb = -0.05
    np.random.seed(1)

    # Verify metric name
    allowed_metrics = ["Statistical parity difference",
                       "Average odds difference",
                       "Equal opportunity difference"]
    if metric_name not in allowed_metrics:
        raise ValueError("Metric name should be one of allowed metrics")

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                     num_class_thresh=100, num_ROC_margin=50,
                                     metric_name=metric_name,
                                     metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)

    fav_inds = dataset_orig_test_pred.scores > thresh_arr[idx]
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    # metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
    #                                   unprivileged_groups, privileged_groups)
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    # metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
    #                                   unprivileged_groups, privileged_groups)

    classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
                                                       dataset_transf_test_pred,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)

    print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr[idx])
    # print("Classification accuracy = %f" % classified_metric_orig_test.accuracy())
    # TPR = classified_metric_orig_test.true_positive_rate()
    # TNR = classified_metric_orig_test.true_negative_rate()
    # bal_acc_test = 0.5 * (TPR + TNR)
    # print("Balanced classification accuracy = %f" % bal_acc_test)
    # print("Disparate impact = %f" % classified_metric_orig_test.disparate_impact())
    # print("Equal opportunity difference = %f" % classified_metric_orig_test.equal_opportunity_difference())
    # print("Average odds difference = %f" % classified_metric_orig_test.average_odds_difference())
    # print("Theil_index = %f" % classified_metric_orig_test.theil_index())
    # print("Statistical parity difference = %f" % classified_metric_orig_test.statistical_parity_difference())
    total_acc[7, iter] = classified_metric_orig_test.accuracy()
    total_balanced_acc[7, iter] = 0.5 * (classified_metric_orig_test.true_positive_rate() +
                                         classified_metric_orig_test.true_negative_rate())
    total_disimpact[7, iter] = classified_metric_orig_test.disparate_impact()
    total_eqopp_diff[7, iter] = classified_metric_orig_test.equal_opportunity_difference()
    total_aveodds_diff[7, iter] = classified_metric_orig_test.average_odds_difference()
    total_theil_idx[7, iter] = classified_metric_orig_test.theil_index()
    stat_parity_diff[7, iter] = classified_metric_orig_test.statistical_parity_difference()

    
    

    # # 9. Prejudice Remover
    # print("#------ 9. Prejudice Remover ------#")
    # model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
    # # thresh_arr = np.linspace(0.01, 0.5, 50)
    #
    # tr_dataset = dataset_orig_train.copy(deepcopy=True)
    # scale = StandardScaler().fit(tr_dataset.features)  # remember the scale
    # tr_dataset.features = scale.transform(tr_dataset.features)
    # model.fit(tr_dataset)
    #
    # valid_dataset = dataset_orig_valid.copy(deepcopy=True)
    # valid_dataset.features = scale.transform(valid_dataset.features)
    # pred_dataset = model.predict(valid_dataset)
    # y_validate_pred_prob = pred_dataset.scores
    #
    # bal_acc_arr = []
    #
    # for thresh in tqdm(thresh_arr):
    #     y_validate_pred = (y_validate_pred_prob[:, 1] > thresh).astype(np.double)
    #
    #     dataset_pred = dataset_orig_valid.copy()
    #     dataset_pred.labels = y_validate_pred
    #
    #     classified_metric = ClassificationMetric(dataset_orig_valid,
    #                                              dataset_pred,
    #                                              unprivileged_groups=unprivileged_groups,
    #                                              privileged_groups=privileged_groups)
    #
    #     bal_acc = 0.5 * (classified_metric.true_positive_rate() + classified_metric.true_negative_rate())
    #     bal_acc_arr.append(bal_acc)
    #
    # thresh_arr_best_ind = np.where(bal_acc_arr == np.max(bal_acc_arr))[0][0]
    # thresh_arr_best = np.array(thresh_arr)[thresh_arr_best_ind]
    #
    # te_dataset = dataset_orig_test.copy(deepcopy=True)
    # te_dataset.features = scale.transform(te_dataset.features)
    # pred_dataset = model.predict(te_dataset)
    #
    # y_data_pred_prob = pred_dataset.scores
    # y_data_pred = (y_data_pred_prob[:, 1] > thresh_arr_best).astype(np.double)
    #
    # dataset_pred = dataset_orig_test.copy()
    # dataset_pred.labels = y_data_pred
    #
    # classified_metric = ClassificationMetric(dataset_orig_test,
    #                                          dataset_pred,
    #                                          unprivileged_groups=unprivileged_groups,
    #                                          privileged_groups=privileged_groups)
    # metric_pred = BinaryLabelDatasetMetric(dataset_pred,
    #                                        unprivileged_groups=unprivileged_groups,
    #                                        privileged_groups=privileged_groups)
    #
    # print("Threshold corresponding to Best balanced accuracy: %6.4f" % thresh_arr_best)
    # # print("Classification accuracy = %f" % classified_metric.accuracy())
    # # TPR = classified_metric.true_positive_rate()
    # # TNR = classified_metric.true_negative_rate()
    # # bal_acc_test = 0.5*(TPR+TNR)
    # # print("Balanced classification accuracy = %f" % bal_acc_test)
    # # print("Disparate impact = %f" % metric_pred.disparate_impact())
    # # print("Equal opportunity difference = %f" % classified_metric.equal_opportunity_difference())
    # # print("Average odds difference = %f" % classified_metric.average_odds_difference())
    # # print("Theil_index = %f" % classified_metric.theil_index())
    # # print("Statistical parity difference = %f" % classified_metric.statistical_parity_difference())
    # total_acc[8, iter] = classified_metric.accuracy()
    # total_balanced_acc[8, iter] = 0.5 * (classified_metric.true_positive_rate() +
    #                                      classified_metric.true_negative_rate())
    # total_disimpact[8, iter] = metric_pred.disparate_impact()
    # total_eqopp_diff[8, iter] = classified_metric.equal_opportunity_difference()
    # total_aveodds_diff[8, iter] = classified_metric.average_odds_difference()
    # total_theil_idx[8, iter] = classified_metric.theil_index()
    # stat_parity_diff[8, iter] = classified_metric.statistical_parity_difference()
    '''
    
    if not os.path.exist('save_compare_data'): os.makedirs('save_compare_data')
    np.savez('save_compare_data/' + name + '_total_acc.npz', total_acc= total_acc)
    np.savez('save_compare_data/' + name + '_total_balanced_acc.npz', total_balanced_acc= total_balanced_acc)
    np.savez('save_compare_data/' + name + '_total_disimpact' + '.npz', total_disimpact = total_disimpact)
    np.savez('save_compare_data/' + name + '_total_eqopp_diff' + '.npz', total_eqopp_diff = total_eqopp_diff)
    np.savez('save_compare_data/' + name + '_total_aveodds_diff' + '.npz', total_aveodds_diff = total_aveodds_diff)
    np.savez('save_compare_data/' + name + '_total_theil_idx' + '.npz', total_theil_idx = total_theil_idx)
    np.savez('save_compare_data/' + name + '_stat_parity_diff' + '.npz', stat_parity_diff = stat_parity_diff)
    np.savez('save_compare_data/' + name + '_total_tpr' + '.npz', total_tpr = total_tpr)
    np.savez('save_compare_data/' + name + '_total_fpr_diff' + '.npz', total_fpr_diff = total_fpr_diff)
    
# print output
print("Classification accuracy:")
print(np.around(np.mean(total_acc, axis=1), decimals=3))

print("Classification accuracy std:")
print(np.around(np.std(total_acc, axis=1), decimals=3))

print("Balanced classification accuracy:")
print(np.around(np.mean(total_balanced_acc, axis=1), decimals=3))

print("Balanced classification accuracy std:")
print(np.around(np.std(total_balanced_acc, axis=1), decimals=3))

print("Disparate impact:")
print(np.around(np.mean(total_disimpact, axis=1), decimals=3))

print("Disparate impact std:")
print(np.around(np.std(total_disimpact, axis=1), decimals=3))

print("Equal opportunity difference:")
print(np.around(np.mean(total_eqopp_diff, axis=1), decimals=3))

print("Equal opportunity difference std:")
print(np.around(np.std(total_eqopp_diff, axis=1), decimals=3))

print("Average odds difference:")
print(np.around(np.mean(total_aveodds_diff, axis=1), decimals=3))

print("Average odds difference std:")
print(np.around(np.std(total_aveodds_diff, axis=1), decimals=3))

print("Theil_index:")
print(np.around(np.mean(total_theil_idx, axis=1), decimals=3))

print("Theil_index std:")
print(np.around(np.std(total_theil_idx, axis=1), decimals=3))

print("Statistical parity difference:")
print(np.around(np.mean(stat_parity_diff, axis=1), decimals=3))

print("Statistical parity difference std")
print(np.around(np.std(stat_parity_diff, axis=1), decimals=3))
print(total_tpr[9])
setting = 'best/' + args.dataset + '/' + name + '/' + str(args.lr) + '_' + str(args.lr_t)
plot(total_acc, 0, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(total_balanced_acc, 1, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(1-total_disimpact), 2, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(total_eqopp_diff), 3, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(total_aveodds_diff), 4, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(total_theil_idx), 5, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(stat_parity_diff), 6, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(total_tpr), 7, dataset_used, setting, method_idx = [10,1,2,3,6,9])
plot(np.abs(total_fpr_diff), 8, dataset_used, setting, method_idx = [10,1,2,3,6,9])

np.savez('save_compare_data/' + name + '_total_acc.npz', total_acc= total_acc)
np.savez('save_compare_data/' + name + '_total_balanced_acc.npz', total_balanced_acc= total_balanced_acc)
np.savez('save_compare_data/' + name + '_total_disimpact' + '.npz', total_disimpact = total_disimpact)
np.savez('save_compare_data/' + name + '_total_eqopp_diff' + '.npz', total_eqopp_diff = total_eqopp_diff)
np.savez('save_compare_data/' + name + '_total_aveodds_diff' + '.npz', total_aveodds_diff = total_aveodds_diff)
np.savez('save_compare_data/' + name + '_total_theil_idx' + '.npz', total_theil_idx = total_theil_idx)
np.savez('save_compare_data/' + name + '_stat_parity_diff' + '.npz', stat_parity_diff = stat_parity_diff)
np.savez('save_compare_data/' + name + '_total_tpr' + '.npz', total_tpr = total_tpr)
np.savez('save_compare_data/' + name + '_total_fpr_diff' + '.npz', total_fpr_diff = total_fpr_diff)