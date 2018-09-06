import numpy as np
import pandas as pd
from konlpy.tag import Twitter
from collections import defaultdict
import time
import pickle

# NaiveBayesClassifier
class NBC:
    def __init__(self, k=1):
        self.k = k
        self.word_probs = defaultdict(tuple)
        # you can change tokenizer to something else.
        self.tokenizer = Twitter()
        self.total_dic, self.pos_dic, self.neg_dic = {}, {}, {}

    def load_data(self, path):
        # load data for training
        data = pd.read_csv(path, sep=',', encoding='CP949')
        sentence = np.array(data)[:, 1:2]  # remove id and remain label
        label = np.array(data)[:, -1]

        return sentence, label

    def tokenizing(self, data, flatten=True):
        # sentence tokenizing
        if flatten:
            data = data.flatten()
        return [self.tokenizer.morphs(token) for token in data]

    def make_dict(self, data, label):
        # make dictionary using defaultdict for probability of words
        token_data = self.tokenizing(data)
        total_dic = defaultdict(int)  # total word dictionary
        pos_dic = defaultdict(int)  # positive word dictionary
        neg_dic = defaultdict(int)  # negative word dictionary

        for i, token in enumerate(token_data):
            # word counting
            for word in token:
                total_dic[word] += 1
                if label[i]:
                    pos_dic[word] += 1
                else:
                    neg_dic[word] += 1

        return total_dic, pos_dic, neg_dic

    def word_prob(self, data, label):
        k = self.k  # for smoothing
        dic = defaultdict(tuple)
        pos = sum([len(token) for token, label in zip(data, label) if label])  # number of pos words
        neg = sum([len(token) for token, label in zip(data, label) if not label])  # number of neg words
        total_dic, pos_dic, neg_dic = self.make_dict(data, label)

        for w in total_dic.keys(): # dictionary = {word : (pos_probs, neg_probs)}
            dic[w] = ((pos_dic[w] + k) / (pos + 2 * k), (neg_dic[w] + k) / (neg + 2 * k))

        return dic

    def class_prob(self, sentence):
        # calculate the probability of positive or negative.
        sentence_token = self.tokenizing(sentence, False)

        pos_prob, neg_prob = 0, 0

        for word in sentence:
            if word in self.word_probs:
                pos_prob += np.log(self.word_probs[word][0])
                neg_prob += np.log(self.word_probs[word][1])

        if (pos_prob, neg_prob) == (0, 0):
            return 0.5

        pos_prob, neg_prob = np.exp(pos_prob), np.exp(neg_prob)
        return pos_prob / (pos_prob + neg_prob)

    def train(self, path, test = False):
        # load data in path and training
        s_t = time.time()
        data, label = self.load_data(path)

        if test:
            data, label = data[:1000], label[:1000]  # small data for test

        print("A total of %d training data." % len(label))
        self.total_dic, self.pos_dic, self.neg_dic = self.make_dict(data, label)
        self.word_probs = self.word_prob(data, label)
        print("Training Completion, Time taken : %.f seconds " % (time.time() - s_t))

    def classify(self, sentence, hard=False):
        # only one sentence classification
        # Hard is only 0(negative) and 1(positive. But, another is 0 to 1
        if hard:
            return np.round(self.class_prob(sentence))

        print(sentence, ':', self.class_prob(sentence))
        return self.class_prob(sentence)

    def doc_classify(self, doc, test=False):
        # doc hard classification
        s_t = time.time()
        test_data, test_label = self.load_data(doc)
        if test:
            test_data, test_label = test_data[:1000], test_label[:1000]  # small data for test
        print("A total of %d test data." % len(test_label))
        test_data = self.tokenizing(test_data)

        predict_label = []
        for sentence in test_data:
            predict_label.append(self.classify(sentence, True))
        print("Testing Completion, Time taken :  %.f seconds ---" % (time.time() - s_t))
        accuracy = len([x for x, y in zip(test_label, predict_label) if x == y]) / len(test_label)
        print('Accuracy :', accuracy)

    def save(self):
        # save parameter
        f = open("./data/save.pkl", 'wb')
        pickle.dump(self.word_probs, f)
        f.close()
        print('Saved completely')

    def load(self):
        # load parameter
        f = open("./data/save.pkl", 'rb')
        self.word_probs = pickle.load(f)
        print('Loaded completely')