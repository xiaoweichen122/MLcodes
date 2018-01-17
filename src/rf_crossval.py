## cross validation for parameter tuning for random forest
## V2 modeling
import os
import sys
#from functions import processfunctions as pf
import numpy as np
import pandas as pd
from datetime import timedelta
import re

from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import pickle

def oob_plot(errors, minv, maxv, xlab=""):
	xs, ys=zip(*errors)
	plt.plot(xs, ys)
	plt.xlim(minv, maxv)
	plt.xlabel(xlab)
	plt.ylabel("OOB error rate")
	plt.show()

def get_datasets(train_id, test_id, dat, cols):
	train_mer = pd.DataFrame(df.ix[train_id, 'MERCHANT_CUSTOMER_ID'])
	test_mer = pd.DataFrame(df.ix[test_id, 'MERCHANT_CUSTOMER_ID'])	
	train_cv =pd.merge(dat, train_mer, on=cols,how='inner')
	test_cv =pd.merge(dat, test_mer, on=cols,how='inner')
	return train_cv, test_cv

def get_thresh(prob, X):
	thrs = np.arange(0.0, 1.01, 0.01)
	np.set_printoptions(precision=3, suppress=True)
	for thr in thrs:
		pred =  (prob[:, 1]>=thr)*1
		confusion = metrics.confusion_matrix(X["LABELS_BIN"], pred)
		precision = metrics.precision_score(X["LABELS_BIN"], pred)
		recall = metrics.recall_score(X["LABELS_BIN"], pred)
		fpr = confusion[0][1] / float(confusion[0][1]+confusion[0][0])
		f1score = metrics.f1_score(X["LABELS_BIN"], pred)
		rr = confusion.flatten()
		rr = np.append(rr, [thr, precision, recall, fpr, f1score])
		if thr == 0:
			output = rr
		else:
			output =np.vstack([output, rr])
	return output

def cv_fitandpred(cvpars, var, df, dat, cols, ncv=5):
	oobs = np.zeros((len(cvpars), ncv+1))
	results = np.zeros((len(cvpars), ncv+1))
	resultstrain = np.zeros((len(cvpars), ncv+1))
	i=0
	j=0
	for par in cvpars:
		print i, par
		rf.set_params(**{var: par})
		j=0
		oobs[i,j] = par
		results[i,j] = par
		resultstrain[i,j] = par
		for train_id, test_id in skf.split(df['MERCHANT_CUSTOMER_ID'], df['LABELS_BIN']):
			j = j+1
			train_cv, test_cv = get_datasets(train_id, test_id, dat, cols)
			rf.fit(train_cv[allcols], train_cv["LABELS_BIN"])
			prob = rf.predict_proba(test_cv[allcols])
			output = get_thresh(prob, test_cv)
			recall = output[np.argmin(abs(output[:,7]-fpr)), 6]
			probtrain = rf.predict_proba(train_cv[allcols])
			outputtrain = get_thresh(probtrain, train_cv)
			recalltrain = outputtrain[np.argmin(abs(outputtrain[:,7]-fpr)), 6]
			error = 1- rf.oob_score_
			print "Test: thr, fpr and recall: ", output[np.argmin(abs(output[:,7]-fpr)), 4].round(4), output[np.argmin(abs(output[:,7]-fpr)), 7].round(4), recall.round(4)
			print "Train: thr, fpr and recall: ", outputtrain[np.argmin(abs(outputtrain[:,7]-fpr)), 4].round(4), outputtrain[np.argmin(abs(outputtrain[:,7]-fpr)), 7].round(4), recalltrain.round(4)
			print "oob error: ", error.round(4)
			oobs[i, j] = error
			results[i, j] = recall
			resultstrain[i, j] = recalltrain
		i=i+1
	return results, resultstrain, oobs

def results_plot(results, resultstrain, oobs, fpr, cv=5, name='maxdepth'):
	means =results[:, 1:(cv+1)].mean(axis=1)
	stds = results[:, 1:(cv+1)].std(axis=1)
	a = np.hstack((results, means[:, None], stds[:, None]))
	np.savetxt("/home/chxiaowe/fpstage2/data/V2/model/cv/cv_results_"+name+".csv", a, fmt='%.4f', delimiter=',', header="par, cv1, cv2, cv3, cv4, cv5, mean, std", comments="")	
	meanstt =resultstrain[:, 1:(cv+1)].mean(axis=1)
	stdstt = resultstrain[:, 1:(cv+1)].std(axis=1)
	b = np.hstack((resultstrain, meanstt[:, None], stdstt[:, None]))
	np.savetxt("/home/chxiaowe/fpstage2/data/V2/model/cv/cv_resultstrain_"+name+".csv", b, fmt='%.4f', delimiter=',', header="par, cv1, cv2, cv3, cv4, cv5, mean, std", comments="")	
	meansoob =oobs[:, 1:(cv+1)].mean(axis=1)
	stdsoob = oobs[:, 1:(cv+1)].std(axis=1)
	c = np.hstack((oobs, meansoob[:, None], stdsoob[:, None]))
	np.savetxt("/home/chxiaowe/fpstage2/data/V2/model/cv/cv_oobs_"+name+".csv", c, fmt='%.4f', delimiter=',', header="par, cv1, cv2, cv3, cv4, cv5, mean, std", comments="")	
	#a = np.genfromtxt("/home/chxiaowe/fpstage2/data/V2/model/cv/cv_results_"+name+".csv", delimiter=',', skip_header=1)
	#b = np.genfromtxt("/home/chxiaowe/fpstage2/data/V2/model/cv/cv_resultstrain_"+name+".csv", delimiter=',', skip_header=1)
	#c = np.genfromtxt("/home/chxiaowe/fpstage2/data/V2/model/cv/cv_oobs_"+name+".csv", delimiter=',', skip_header=1)
	fig = plt.figure()
	plt.errorbar(a[:, 0], a[:, (cv+1)], a[:, (cv+2)], marker='^', ecolor='c', color ='b' )
	plt.errorbar(b[:, 0], b[:, (cv+1)], b[:, (cv+2)], marker='s', ecolor='y', color='r')
	plt.errorbar(c[:, 0], c[:, (cv+1)], c[:, (cv+2)], marker='.', ecolor='k', color='k')
	axes = plt.gca()
	ymax = max(list(a[:,(cv+1)]+a[:,(cv+2)]) + list(b[:,(cv+1)]+b[:,(cv+2)]) + list(c[:,(cv+1)]+c[:,(cv+2)]))
	axes.set_ylim([0,ymax+0.05])
	plt.xlabel(name, fontsize=18)
	plt.ylabel('recall/ooberr at fpr=' + str(fpr), fontsize=16)
	#plt.show()
	fig.savefig('/home/chxiaowe/fpstage2/data/V2/model/cv/cv_results_'+name+'.png') # Use fig. here
	return None



## load data and feature list
foldername = "/home/chxiaowe/fpstage2/data/V2/features/"
dat = pd.read_pickle(foldername+"features_processed_data")
## feature set
allcols = pd.read_pickle(foldername + 'features_processed_list')
allcols = list(allcols.FEATURES)


# ## training/testing set
# trainfilename = "/home/chxiaowe/fpstage2/data/V2/defect/train_indexid_seller.tsv"
# train_index = pd.read_table(trainfilename, sep='\t',  header=None, names=["INDEX_ID"])
# # testfilename = "/home/chxiaowe/fpstage2/data/V2/defect/test_indexid_seller.tsv"
# # test_index = pd.read_table(testfilename, sep='\t',  header=None, names=["INDEX_ID"])
# traindat = pd.merge(dat, train_index, how="inner", on="INDEX_ID")
# # testdat = pd.merge(dat, test_index, how="inner", on="INDEX_ID")


## cross validation sets
cols = ["MERCHANT_CUSTOMER_ID"]
df = dat[['MERCHANT_CUSTOMER_ID', 'LABELS_BIN']].drop_duplicates(cols, keep='first')
df = df.reset_index(drop=True)
skf = StratifiedKFold(n_splits=5, random_state=33, shuffle=True)
skf.get_n_splits(df['MERCHANT_CUSTOMER_ID'], df['LABELS_BIN'])
fpr = 0.015


## max_depth
rf = ensemble.RandomForestClassifier( max_depth=None,random_state = 33, min_samples_split=20, oob_score=True, n_estimators=120, max_features=80, min_samples_leaf=20)
var = 'max_depth'
cvpars = range(3, 61, 5)
results, resultstrain, oobs = cv_fitandpred(cvpars, var, df, dat, cols, ncv=5)	
results_plot(results, resultstrain, oobs, fpr, cv=5, name=var)

## n_estimators
rf = ensemble.RandomForestClassifier( max_depth=None,random_state = 33, min_samples_split=20, oob_score=True, n_estimators=120, max_features=80, min_samples_leaf=20)
var = 'n_estimators'
cvpars = range(10, 200, 20)
results, resultstrain, oobs = cv_fitandpred(cvpars, var, df, dat, cols, ncv=5)	
results_plot(results, resultstrain, oobs, fpr, cv=5, name=var)

## min_samples_split
rf = ensemble.RandomForestClassifier( max_depth=None,random_state = 33, min_samples_split=20, oob_score=True, n_estimators=120, max_features=80, min_samples_leaf=20)
var = 'min_samples_split'
cvpars = range(2, 200, 20)
results, resultstrain, oobs = cv_fitandpred(cvpars, var, df, dat, cols, ncv=5)	
results_plot(results, resultstrain, oobs, fpr, cv=5, name=var)

## min_samples_leaf
rf = ensemble.RandomForestClassifier( max_depth=None,random_state = 33, min_samples_split=20, oob_score=True, n_estimators=120, max_features=80, min_samples_leaf=20)
var = 'min_samples_leaf'
cvpars = range(2, 200, 20)
results, resultstrain, oobs = cv_fitandpred(cvpars, var, df, dat, cols, ncv=5)	
results_plot(results, resultstrain, oobs, fpr, cv=5, name=var)

## max_features
rf = ensemble.RandomForestClassifier( max_depth=None,random_state = 33, min_samples_split=20, oob_score=True, n_estimators=120, max_features=80, min_samples_leaf=20)
var = 'max_features'
cvpars = range(10, 200, 20)
results, resultstrain, oobs = cv_fitandpred(cvpars, var, df, dat, cols, ncv=5)	
results_plot(results, resultstrain, oobs, fpr, cv=5, name=var)







