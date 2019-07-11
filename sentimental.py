import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# dictionary is used in order to avoid repeatation
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))

# 800 of 10000 is used for training an drest 200 is used for checking the accuracy
train_set = neg_reviews[:800] + pos_reviews[:800]
test_set =  neg_reviews[800:1000] + pos_reviews[800:1000]
print(len(train_set),  len(test_set))

classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy * 100)

#user can enter a review
review=raw_input("Enter a movie review : ")
print(review)

words = word_tokenize(review_santa)
words = create_word_features(words)

# to print whether the review is positive / negative
print(classifier.classify(words))
