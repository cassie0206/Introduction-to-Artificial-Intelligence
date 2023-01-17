import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm



class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        model = {}
        unimodel = {}
        features = []
        #print('get ngram')
        for tokens in corpus_tokenize:
            
            for i in range(1, len(tokens) - 1):
                features += [(tokens[i],tokens[i+1])]
                if tokens[i] in model:
                    unimodel[tokens[i]] +=1
                    if tokens[i + 1] in model[tokens[i]]:
                        model[tokens[i]][tokens[i+1]] += 1
                    else:
                        model[tokens[i]][tokens[i+1]] = 1
                else:
                    unimodel[tokens[i]] =1
                    model[tokens[i]] = {}
                    model[tokens[i]][tokens[i+1]] = 1
                    
        for tokens in model:
            num = unimodel[tokens]
            #for token in model[tokens]:
                #num += model[tokens][token]
            for token in model[tokens]:
                model[tokens][token] /= num
           
        return model, features
    

    
        # end your code
  
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)
        #self.model = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        
        # begin your code (Part 2)
        #print('compute perplexity')
        total = 0
        num = 0
        #print('f')
        
        for text in corpus:            
            for i in range(len(text) - 1):
                num += 1
                if text[i] in self.model:
                    if text[i+1] in self.model[text[i]]:
                        pr = self.model[text[i]][text[i+1]]                    
                        total += (math.log(pr, 2))
                    else:
                        pr = 1 / len(self.model[text[i]])
                        total += (math.log(pr, 2))
                        
        #print(-(total/num))
        perplexity = math.pow(2, -(total/num))
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        #print('sentimental1')
        feature_num = 200
        features = Counter(self.features)
        features = sorted(features.items(),key = lambda x:x[1],reverse=True)
        feature_num = min(feature_num, len(features))
        #features = list(features.keys())
        features = features[:feature_num]

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        train_corpus_embedding = []
        
        train_token = [['[CLS]'] + self.tokenize(document) for document in df_train['review']] 
        #print(train_token)
        
        #print('sentimental2')
        for text in train_token:
            embedded = []
            bi_train = []
            for i in range(len(text) - 1):
                #print(text[i])
                bi_train += [(text[i], text[i+1])]
            
            for feature in features:
                if feature[0] in bi_train:
                    embedded += [bi_train.count(feature[0])]
                else:
                    embedded += [0]
            #print(embedded)
            train_corpus_embedding.append(embedded)
            
        train_corpus_embedding = np.array(train_corpus_embedding)
        #print(train_corpus_embedding)
                    
        #print('sentimental3')   
        test_corpus_embedding = []
        
        test_token = [['[CLS]'] + self.tokenize(document) for document in df_test['review']] 
        
        for text in test_token:
            embedded = []
            bi_test = []
            for i in range(len(text) - 1):
                bi_test += [(text[i], text[i+1])]
            
            for feature in features:
                if feature[0] in bi_test:
                    embedded += [bi_test.count(feature[0])]
                else:
                    embedded += [0]
                
            test_corpus_embedding.append(embedded)
            
        
        test_corpus_embedding = np.array(test_corpus_embedding)
        #print(test_corpus_embedding)
        
        df_train['sentiment'] = np.array(df_train['sentiment'])
        df_test['sentiment'] = np.array(df_test['sentiment'])
        #print('sentimental fead')
        # end your code
        
        

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
    
