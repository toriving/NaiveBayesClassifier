import os
from NaiveBayesClassfier import NBC

TRAINING_FILE = './data/ratings_train.csv'
TEST_FILE = './data/ratings_test.csv'
SAVE_FILE = './data/save.pkl'

def main():
  classifier = NBC()
  if not os.path.exists(SAVE_FILE):
    classifier.train(TRAINING_FILE)
    classifier.save()
  else:
    classifier.load()
  classifier.classify('♥')
  classifier.classify('너무너무싫다')
  classifier.doc_classify(TEST_FILE)

if __name__ == '__main__':
  main()
