#!/usr/bin/env python3


import os
import csv
import sys
import codecs
import re
import pandas as pd 
from pandas.io.json import json_normalize
import numpy as np
import argparse
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.tokenizer import Tokenizer
#from spacy.vectors import Vectors
import json
import random
from sklearn_pandas import DataFrameMapper
import collections


from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest,f_classif,chi2
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD,PCA

from sklearn.externals import joblib

parser = argparse.ArgumentParser(description = 'Classify users into countries')
parser.add_argument('jsondb',help='the json file with at least user:location, tweets')
parser.add_argument('--users', type=int, help='if None or too high uses all the users available')
parser.add_argument('result_filename',type=str ,help='filename where the results should be stored')
arguments = parser.parse_args()

jsondb = arguments.jsondb
users =arguments.users
result_filename = arguments.result_filename

vectorizer = TfidfVectorizer()
le = LabelEncoder()

def main():
	print("...started classify.py")
	nlp = spacy.load('en')

	print("...reading the json file")


	#load in db with user and tweets
	with codecs.open(jsondb,'r') as json_data:
		data_raw = json.load(json_data)

	test_params = [(700,35)]

	results = []

	outfile = codecs.open('resultfiles/'+result_filename+'.txt','w','utf-8')
	outfile.write('overall_accuracy\tuser_accuracy\n')
	
	for (num_tweets,size_batches) in test_params:


		data, groups = prep_tweets(data_raw,nlp,num_tweets,size_batches,users)

		print('classify now')
		overall_accuracy, average_user_accuracy, data_distribution = classifier(data,groups,result_filename)
		outfile.write(str(overall_accuracy))
		outfile.write('\t')
		outfile.write(str(average_user_accuracy))
		outfile.write('\n')
		outfile.write(str(data_distribution))
		outfile.write('\n')
		

		print(num_tweets,size_batches)


	print('done!')



#https://stackoverflow.com/questions/34710281/use-featureunion-in-scikit-learn-to-combine-two-pandas-columns-for-tfidf

class DataFrameColumnExtracter(TransformerMixin):

	def __init__(self, column):
		self.column = column
	
	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return X[self.column]

def tokenize_tweets(nlp,tweet,tokenizer):
	token_string = nlp(tweet)

	tokens = [str(item.text) for item in token_string]
	lemmas = [str(item.lemma_) for item in token_string]

	return tokens,lemmas
		

def predict_user(test_text,test_y,pipeline):
	row_1 = test_text.iloc[0].name
	user_ind = row_1.split('_')
	user_ind = user_ind[0]
	#print('us_ind',user_ind)
	user_loc = []
	i = 0

	over_all_acc = []

	for row in test_text.itertuples():

		temp_ind = row[0].split('_')
		temp_ind = temp_ind[0]
		
		if user_ind == temp_ind:
			user_loc.append(str(row[0]))
		else:
			
			user_test = test_text.loc[user_loc[:]]
			user_y = test_y[i:i+len(user_loc)]
			prediction = pipeline.predict(user_test)
			accuracy = accuracy_score(user_y, prediction)
			counter = collections.Counter(prediction)
			
			i += len(user_loc)

			user_ind = temp_ind
			user_loc = [row[0]]

			over_all_acc.append(accuracy)

			
	print ('over_all_acc',sum(over_all_acc)/len(over_all_acc))

	return over_all_acc


def remove_RT_and_sample(user_tweets,n):
	non_RTs = [x for x in user_tweets if x.startswith("'RT") == False]
	if n <= len(non_RTs):
		sample_tweets = random.sample(non_RTs,n)
		return sample_tweets
	else:
		return non_RTs


#function to handle the tweets
def prep_tweets(data,nlp,num_tweets, size_batches,users):
	print('...preparing tweets')


	tweet_keys = list(data.keys())
	if  users >= len(tweet_keys) or users == None:
		tweet_keys = list(data.keys())
	else:
		tweet_keys = random.sample(list(data.keys()),users)

	print('number of users:',len(tweet_keys))
	if 'lang' in tweet_keys:
		tweet_keys.remove('lang')
	classify_data = {}

	tokenizer = Tokenizer(nlp.vocab)

	i = 1
	user_mapping = {}
	
	groups = []
	at_dict = {}

	for user in tweet_keys:
		
		user_mapping[i] = user
		j = 1
		sample_tweets = remove_RT_and_sample(data[user]['tweets'],num_tweets)
		tweet_batch = []

		for tweet in sample_tweets:
			#create a unique id for every tweet 

			if j % size_batches == 0:
				tweet_batch.append(tweet.strip("'"))

				tweet = (" ").join(tweet_batch)
				#print(tweet)

				index = str(i)+'_'+str(j)
				classify_data[index] = {}
				
				tweet_tok,lemmas = tokenize_tweets(nlp,tweet,tokenizer)

				classify_data[index]['country'] = data[user]['country']
				classify_data[index]['location'] = data[user]['location']
				classify_data[index]['user_name'] = user
			
				classify_data[index]['lemmas'] = lemmas
				
				groups.append(i)
				tweet_batch = []
			else:
				tweet_batch.append(tweet)

				
			j += 1


		i += 1
		if i % 10 == 0:
				print('...',i)

	with codecs.open('user_data.json','w','utf-8') as user_dump:
		json.dump(classify_data,user_dump)
	
	return classify_data, groups
#https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
	''' Get top n tfidf values in row and return them with their corresponding feature names.'''
	topn_ids = np.argsort(row)[::-1][:top_n]
	top_feats = [(features[i], row[i]) for i in topn_ids]
	df = pd.DataFrame(top_feats)
	df.columns = ['feature', 'tfidf']
	return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
	''' Top tfidf features in specific document (matrix row) '''
	row = np.squeeze(Xtr[row_id].toarray())
	return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
	''' Return the top n features that on average are most important amongst documents in rows
		indentified by indices in grp_ids. '''
	if grp_ids:
		D = Xtr[grp_ids].toarray()
	else:
		D = Xtr.toarray()

	D[D < min_tfidf] = 0
	tfidf_means = np.mean(D, axis=0)
	return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
	''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
		calculated across documents with the same class label. '''
	dfs = []
	labels = np.unique(y)
	for label in labels:
		ids = np.where(y==label)
		feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
		feats_df.label = label
		dfs.append(feats_df)
	return dfs


def plot_tfidf_classfeats_h(dfs,result_filename):
	''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
	x = np.arange(len(dfs[0]))
	for i, df in enumerate(dfs):
		#print(df.label)
		filename = 'feature_csvs/'+df.label+result_filename
		df.to_csv(filename)

def print_top25(vectorizer, clf, class_labels):
	"""Prints features with the highest coefficient values, per class"""
	feature_names = vectorizer.get_feature_names()
	for i, class_label in enumerate(class_labels):
		top10 = np.argsort(clf.coef_[i])[-25:]
		print("%s: %s" % (class_label,
			 " ".join(feature_names[j] for j in top10)))	
		
#https://gist.github.com/zacstewart/5978000
def classifier(data, groups,result_filename):
	print("...start classifier")

	data = pd.DataFrame.from_dict(data,orient='index')

	mapper = DataFrameMapper([
		('country',None),
		('lemmas',None),
		('user_name',None)
		],
		df_out=True)
	data = mapper.fit_transform(data)

	k_fold = GroupKFold(n_splits=10)
	i = 1
	print(len(data))

	transformer = [

			('pipeline_lemma', Pipeline([
				('selector',DataFrameColumnExtracter('lemmas')),
				('tfidif',TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,1)))
				]))
			
			]

	pipeline_MultinomaialNB = Pipeline([
			('union', FeatureUnion(

			transformer_list = transformer

			)),
			('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))
			])

	
	pipeline_lemma = Pipeline([
				('selector',DataFrameColumnExtracter('lemmas')),
				('tfidf',TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,1)))
				])


	pipelines = [('multinomial',pipeline_MultinomaialNB)] 
	
	user_accs = []
	all_acc = []
	
	for (name,pipeline) in pipelines:
		predictions = []
		print(name)
		 
		groups = np.asarray(groups)
		for train_indices, test_indices in k_fold.split(data,groups=groups):

			train_text = data.iloc[train_indices]
			train_y = data.iloc[train_indices]['country'].values.astype(str)
			train_text = train_text.drop('country',axis=1)

			test_text = data.iloc[test_indices]
			test_y = data.iloc[test_indices]['country'].values.astype(str)
			test_text = test_text.drop('country',axis=1)

			print('fitting...')
			pipeline.fit(train_text,train_y)

			

			prediction = pipeline.predict(test_text)

			over_all_user_acc = predict_user(test_text,test_y,pipeline)
			accuracy = accuracy_score(test_y, prediction)
			print(i, accuracy)

			#save model per iteration to choose best iteration in the end
			filename = 'models/' + result_filename + name + str(i) + str(accuracy) + '.sav'
			joblib.dump(pipeline, filename)
			predictions.append(accuracy)
		
			i += 1


		user_accs += over_all_user_acc
		average_acc = sum(predictions)/len(predictions)
		all_acc.append(average_acc)
		print('average accuracy',name, sum(predictions)/len(predictions))
	
	print('...plotting')
	y = data['country'].values.astype(str)
	classes = list(set(y))
	user_names = data['user_name'].values.astype(str)

	user_accuracy = sum(user_accs)/len(user_accs)

	print('average_user_acc', user_accuracy)
	print('average_all', all_acc[0])

	print(set(data['country'].values))
	counter = collections.Counter(data['country'].values)
	data_distribution = counter.most_common()
	print(counter.most_common())
	

	print('...done!')
	return all_acc, user_accs, data_distribution

if __name__ == '__main__':
	main()




