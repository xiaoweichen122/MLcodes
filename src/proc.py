import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import re
import pickle

def monthlydiffcal(dat, wcols, prefix, weeks, fillna=0):
	dat[wcols] = dat[wcols].fillna(fillna)
	wcolnames = set([s.rsplit('_', 1)[0] for s in wcols])
	filename = "/home/chxiaowe/fpstage2/data/V2/features/trans_monthlydiff"
	if not os.path.exists(filename):
		monthtrans = pd.DataFrame(columns=['PREFIX', 'WCOLNAMES', 'WEEKS'])
		monthtrans.to_pickle(filename)
	monthtrans = pd.read_pickle(filename)
	wwcc = ','.join(wcolnames)
	wwkk = ','.join(weeks)
	tmp = pd.DataFrame([[prefix, wwcc, wwkk]], columns=['PREFIX', 'WCOLNAMES', 'WEEKS'])
	if not ((monthtrans['PREFIX']==prefix) & (monthtrans['WCOLNAMES']==wwcc) & (monthtrans['WEEKS']==wwkk)).any():
		monthtrans = pd.concat([monthtrans, tmp], ignore_index=True)
		monthtrans.to_pickle(filename)
	print "Monthly difference calculation..."
	for s in wcolnames:
		print s
		dat[s+'_30_VALUE'] = dat[s+'_30']
		for w in range(1, len(weeks)):
			### need to calculate each month first and then the difference
			###############
			dat[s+'_'+weeks[w]+'_VALUE'] = dat[s+'_'+weeks[w]]-dat[s+'_'+weeks[w-1]]
			dat[s+'_'+weeks[w]+weeks[w-1]+"_DIFF_FE"] = dat[s+'_'+weeks[w]+'_VALUE']-dat[s+'_'+weeks[w-1]+'_VALUE']
			dat[s+'_'+weeks[w]+weeks[w-1]+"_DIFFPROP_FE"] = dat[s+'_'+weeks[w]+weeks[w-1]+"_DIFF_FE"]/dat[s+'_'+weeks[w-1]+'_VALUE'].astype(float)
			dat.ix[dat[s+'_'+weeks[w-1]+'_VALUE']==0, s+'_'+weeks[w]+weeks[w-1]+"_DIFFPROP_FE"] = 0
	return dat
	
## remove columns with many missing values or single value column
def rm_missingsingle(X, cols, featurecols, mnum = 200, misvalue = 0):
	## perc: missing percentage cutoff
	## fillna: fill missing values
	if type(cols) is str:
		cols = [cols]
	if len(cols)==0:
		print "No column to process..."
		return X, featurecols
	if (X[cols].isnull().sum(axis=0).sum()) > 0:
		sys.exit("NaN exists. Fill in missing value before calling this function...") 
	nrow = float(X.shape[0])
	i=0
	for cc in cols:
		flag = 0
		tmp = X[cc].copy()
		mis = sum(tmp==misvalue)
		uniqcc = list(tmp.unique())
		if misvalue in uniqcc:
			uniqcc.remove(misvalue)
		if (len(uniqcc) == 1) | (len(uniqcc) == 0):
			flag = 1
			valueflag = len(uniqcc)
		if (nrow-mis) < mnum:
			flag = 2
			valueflag = (nrow-mis)
		# if (len(uniqcc) > 1) & (str(X[cc].dtype)=='object'):
		# 	vcc = tmp.value_counts()
		# 	domi = vcc[0]
		# 	if vcc.index.values[0] == misvalue:
		# 		domi = vcc[1]
		# 	if (nrow - domi - mis) < mnum:
		#		flag = 3
		if flag != 0:
			i=i+1
			print "remove: ", cc, ", flag: ", flag, ", value ", valueflag
			X = X.drop(cc, axis=1)
			featurecols.remove(cc)
	print '# of removed cols: ', i
	return X, featurecols


## categorical to dummy
def categorical_dummy(X, cols):
	if len(cols)==0:
		return X
	print cols
	X = pd.get_dummies(X, columns=cols)
	return X

## categorical to weight of evidence
def categorical_woe(X, cols, label="LABELS_BIN"):
	# calculate adjusted woe.
	# num: # of counts for one level. if smaller than num, bin these categories together
	# label: column name of labels
	sum0 = sum(X[label]==0)
	sum1 = sum(X[label]==1)
	woedict = {}
	for col in cols:
		print col
		## count table
		table0 = X[X[label]==0].groupby([col]).size().reset_index(name="cnt0")
		table1 = X[X[label]==1].groupby([col]).size().reset_index(name="cnt1")
		table0['cnt0'] = table0['cnt0']+0.5
		table1['cnt1'] = table1['cnt1']+0.5
		tt = pd.merge(table0, table1, how="outer", on=col)
		tt = tt.fillna(0.5)
		tt["div0"] = tt['cnt0']/sum0
		tt["div1"] = tt['cnt1']/sum1
		tt["woe"] = 0
		tt["woe"] = tt["div1"]/tt["div0"]
		tt["woe"] = np.log(tt["woe"])
		uniq = X[col].unique()
		X[col+"_woe"] = 0.0
		for s in uniq:
			X.loc[X[col]==s, col+"_woe"] = tt.loc[tt[col]==s, "woe"].values[0]
		X.drop(col, inplace=True, axis=1)
		woedict[col] = tt[[col, "woe"]]
	return X, woedict

## transform categorical data
def categorical_transform(X, cols, num1=10, num2=50, name="rrother", label="LABELS_BIN"):
	## num1: # of levels in variable. If more than num, use woe; otherwise, use dummy
	## num2: binning rare levels. number to define rare.
	## name: name for binned level
	## label: label in categorical_woe
	dummycols = []
	woecols = []
	rrothers = {}
	for col in cols:
		## strip all whitespace characters from start and end
		#X[col]=X[col].str.strip()
		## binning infrequent levels (count <=num2)
		if X[col].isnull().sum()>0:
			print col, " has NaN values"
		## remove all non-alphanumeric characters
		## change the dummy variable to lower case
		X[col]=map(lambda x:x.lower(), X[col])
		X[col]=X[col].str.replace(r"\W+", ".")
		cnt = X[col].value_counts()
		rrothers[col] = list(cnt[cnt<=num2].index) 
		X.loc[X[col].isin(cnt[cnt<=num2].index), col] = name
		print col, ": ", len(X[col].unique())
		if len(X[col].unique())<=num1:
			dummycols.append(col)
		else:
			woecols.append(col)
	print "Categorical to dummy..."
	columnvalues = dict([(cc, list(X[cc].unique())) for cc in dummycols])
	X = categorical_dummy(X, dummycols)	
	print "Categorical to WOE..."
	X, woedict = categorical_woe(X, woecols, label=label)
	return X, columnvalues, woedict, rrothers

