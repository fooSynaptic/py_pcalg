import codecs
import sys
#import gensim
import logging
from gensim import corpora, models, similarities
import jieba
import numpy as np
import pandas as pd
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#the source corpus path is ~/Downloads/cnews/cnews.train.txt
def preprocess_data(f_text, f_stopwords, slic):
	#we formate the preprocess then return the 
	doc = open(f_text).readlines()
	documents = [d.strip().split('\t')[1] for d in doc][:slic]


	# remove common words and tokenize
	with open(f_stopwords) as f:
		stoplist = [x.strip() for x in f.readlines()]

	texts = [[word for word in jieba.cut(document) if word not in stoplist]\
	for document in documents]

	#remove words that appear only once
	from collections import defaultdict
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1

	texts = [[token for token in text if frequency[token] > 1]\
	for text in texts]

	from pprint import pprint
	#pprint(texts[:3])

	dictionary = corpora.Dictionary(texts)
	#dictionary.save('/tmp/deerwester.dict')

	n_token = len(dictionary.token2id)
	print('we have total token with number of :', len(dictionary.token2id))

	'''
	id2token = dict()
	for k, v in zip(dictionary.token2id.keys(), dictionary.token2id.values()):
		id2token.setdefault(v, k)
	'''


	corpus = [dictionary.doc2bow(text) for text in texts]

	tfidf = models.TfidfModel(corpus)
	return tfidf, n_token, texts, dictionary
# save as file
#print(tfidf, type(tfidf), dir(tfidf))
#corpora.MmCorpus.serialize('/tmp/corpus.mm', tfidf)

def main():
	parser = argparse.ArgumentParser(description = 'add argument to the program!')
	parser.add_argument("--f_text", help = "raw corpus file")
	parser.add_argument("--stop_text", help = "stopwords file")
	parser.add_argument("--slic", help = "slice the rawcorpus or ':' for all")

	#print(parser.parse_args())
	f_text, stopwords_path, slic = parser.parse_args()

	if not f_text or not stop_text or not slic:
		doc_path = '/Users/ajmd/Downloads/cnews/cnews.test.txt'
		stopwords_path = '/Users/ajmd/data/stopwords/CNstopwords.txt'
		slic = 500
	else:
		doc_path = f_text
		stopwords_path = stop_text
		slic = slic

	tfidf, n_token, texts, dictionary = preprocess_data(doc_path, stopwords_path, slic)
	# we revector the text in the raw file
	mshape = (tfidf.num_docs, n_token)
	real_matrix = np.zeros(shape = mshape)
	for i in range(len(texts)):
		real_vec = real_matrix[i,]
		#print(c)
		vec = dictionary.doc2bow(texts[i])
		#print('vec:{}'.format(vec))
		vector = tfidf[vec]
		#print('vector:', vector)
		for t in vector:
			idx, val = t
			#print(idx, val)
			real_vec[idx] = val
			
		#print(i, ':', real_vec, real_vec.sum())	
		#break
	#print(real_matrix)
	print('saving to csv format...with shape of {}'.format(real_matrix.shape))
	header = np.array(list(dictionary.token2id.keys())).reshape(-1,8178)
	#print(header)
	print("Shape of header:".format(header.shape))
	final_array = np.vstack((header, real_matrix))
	data = pd.DataFrame(final_array)
	return data
	#data.to_csv('./data1.csv',encoding = 'utf-8')
	#np.savetxt("./text_feature.csv", final_array, delimiter=',')
		

if __name__ == "__main__":
	main()
