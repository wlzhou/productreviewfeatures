from __future__ import division
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import tfidfSparse as tf
from scipy.sparse import csr_matrix, coo_matrix
import lsi
import n_gram as bigram
import itertools

''' This module contains functions for SVR regression as well as
    related utility methods (i.e. compiling the targets vector,
	normalizing features, and training/testing using support vector
	regression.  '''

def compile_targets(filename, threshold = 0.6, num = 0):
	# Construct target vector
	ratings = []
	with open(filename, 'r') as f:
		line = f.readline()
		while line != '':
			if 'review/helpfulness: ' in line:
				line = line[len('review/helpfulness: '):]
				numbers = line.split('/')
				percent = float(numbers[0]) / float(numbers[1])
				if percent > threshold and float(numbers[1]) >= num:
					ratings.append(1)
				else:
					ratings.append(0)
			line = f.readline()

	return np.array(ratings)

def normalize_features(X):
	np.normalize(X)

def train(X, y):
	svm = SVC()
	svm.fit(X, y)
	return svm

def test(svm, X):
	return svm.predict(X)

# converts a list of dictionaries to a 
# scipy sparse CSR matrix, given a key id map
# If keys are already integers, please
# set the length of the ids
def dictListToCSR(listDict, keyIdMap = None, idLen = None):
	# Create the appropriate format for the COO format.
	assert(keyIdMap != None or idLen != None)
	featureLength = len(listDict)
	data = []
	i = []
	j = []

	if idLen == None: idLen = len(keyIdMap)
	if keyIdMap == None:
		keyId = lambda x: x
	else:
		keyId = lambda x: keyIdMap[x]

	# A[i[k], j[k]] = data[k]
	for x in range(featureLength):
		for key in listDict[x]:
			i.append(x)
			j.append(keyId(key))
			data.append(listDict[x][key])

	# Create the COO-matrix
	coo = coo_matrix((data, (i, j)), shape = (featureLength, idLen))
	return csr_matrix(coo, dtype = np.float64)

# maps all keys in an iterable of dictionaries to integer id's so
# the list of dictionaries can be converted to sparse CSR format 
def getKeyIds(listDict):

	allKeys = set()
	for dictionary in listDict:
		for key in dictionary:
			allKeys.add(key)
	allKeys = list(allKeys)
	keyIdMap = {}
	for i in range(len(allKeys)):
		keyIdMap[allKeys[i]] = i

	return keyIdMap


if __name__ == "__main__":
	""" General Testing Paramaters """
	thres = .6
	num = 5
	y = compile_targets('random_train.txt', threshold = thres, num = num)
	actual = compile_targets('random_test.txt', threshold = thres, num = num)
	# """ tf-idf """
	# X, idsLength = tf.tfidf('random_train2.txt')
	# X = dictListToCSR(X, idLen = idsLength)
	# # Xp = tf.tfidf('random_test.txt')
	# svm = train(X, y)
	# predictions = test(svm, X)

	# predCount = np.bincount(predictions)
	# actualCount = np.bincount(actual)
	# comparedCount = np.bincount(predictions + actual)
	# if len(predCount) == 1: predCount = np.append(predCount, 0)
	# if len(comparedCount) == 2: comparedCount = np.append(comparedCount, 0)
	# precision = comparedCount[2] / predCount[1]
	# recall = comparedCount[2] / actualCount[1]
	# accuracy = (comparedCount[0] + comparedCount[2]) / len(predictions)
	# print "Tf-idf Testing: "
	# print "Accuracy: " + "{:.2%}".format(accuracy)
	# print "Precision: " + "{:.2%}".format(precision)
	# print "Recall: " + "{:.2%}".format(recall)

	''' Bigram Testing: perplexities only'''
	model = bigram.ClassifyEssays(lamb = .03, eps = 1e-6)
	model.buildNgram("random_train2.txt", threshold = thres, num = num)
	perplexities, bigrams = model.classify("random_train.txt")
	testPerplexities, testBigrams = model.classify("random_test.txt")

	X = perplexities
	Xp = testPerplexities
	svm = train(X, y)
	predictions = test(svm, Xp)

	predCount = np.bincount(predictions)
	actualCount = np.bincount(actual)
	comparedCount = np.bincount(predictions + actual)
	print predCount
	print actualCount
	if len(predCount) == 1: predCount = np.append(predCount, 0)
	if len(comparedCount) == 2: comparedCount = np.append(comparedCount, 0)
	precision = comparedCount[2] / predCount[1]
	recall = comparedCount[2] / actualCount[1]
	accuracy = (comparedCount[0] + comparedCount[2]) / len(predictions)
	print "Bigram Testing: perplexities only: "
	print "Accuracy: " + "{:.2%}".format(accuracy)
	print "Precision: " + "{:.2%}".format(precision)
	print "Recall: " + "{:.2%}".format(recall)

	''' Bigram Testing: full bigram vectors'''
	keyIdMap = getKeyIds(itertools.chain(bigrams, testBigrams))
	X = dictListToCSR(bigrams, keyIdMap = keyIdMap)
	Xp = dictListToCSR(testBigrams, keyIdMap = keyIdMap)

	svm = train(X, y)
	predictions = test(svm, Xp)

	predCount = np.bincount(predictions)
	actualCount = np.bincount(actual)
	print predCount
	print actualCount
	comparedCount = np.bincount(predictions + actual)
	if len(predCount) == 1: predCount = np.append(predCount, 0)
	if len(comparedCount) == 2: comparedCount = np.append(comparedCount, 0)
	precision = comparedCount[2] / predCount[1]
	recall = comparedCount[2] / actualCount[1]
	accuracy = (comparedCount[0] + comparedCount[2]) / len(predictions)
	print "Bigram Testing: full bigram vectors: "
	print "Accuracy: " + "{:.2%}".format(accuracy)
	print "Precision: " + "{:.2%}".format(precision)
	print "Recall: " + "{:.2%}".format(recall)








