import pandas as pd
import re
import stopwords_guilannlp as sg
import math
import numpy as np
from scipy.optimize import fmin_cg
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import multilabel_confusion_matrix


pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def preprocessing(text):
    cleared_text = []
    text = re.sub('[۰۱۲۳۴۵۶۷۸۹]', '', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub('[$!?/,.%+*_^():{}"؟><]', '', text)
    text = re.sub("-", '', text)
    text = re.sub("'", '', text)
    text = re.sub('قیمت', '', text)
    text = re.sub('فروش', '', text)
    text = re.sub('خرید', '', text)
    text = re.sub('خریدار', '', text)
    text = re.sub('فروشنده', '', text)
    text = re.sub('تعداد', '', text)
    text = re.sub('عدد', '', text)
    text = re.sub('معاوضه', '', text)
    text = re.sub('سالم', '', text)
    text = re.sub('نو', '', text)
    text = re.sub('تخفیف', '', text)
    text = text.split()
    stopwords = sg.stopwords_output("Persian", "nar")
    for w in text:
        if w not in stopwords:
            if len(w) > 2:
                cleared_text.append(w)
    cleared_text = ' '.join(cleared_text)
    return cleared_text


def tf_computation(document, bag_of_words):
    tf_doc = {}
    bow_count = len(bag_of_words)
    # print(bow_count)
    for w, count in document.items():
            tf_doc[w] = float(count / (bow_count + 1))
    return tf_doc


def idf_computation(docs):
    n = len(docs)
    idf_dict = dict.fromkeys(docs[0].keys(), 0)
    for document in docs:
        for w, val in document.items():
            if val > 0:
                idf_dict[w] += 1
    for w, val in idf_dict.items():
        try:
            idf_dict[w] = math.log(n / float(val))
        except:
            continue
    return idf_dict


def tf_idf_computation(tf, idfs):
    tf_idf = {}
    for w, val in tf.items():
        tf_idf[w] = val * idfs[w]
    return tf_idf


def tf_idf(clean_doc, BoW_train):
    BoW = []
    for row in clean_doc:
        BoW.append(row.split())
    # print(BoW)
    unique_words = []
    for l in BoW_train:
        unique_words = set(unique_words).union(set(l))
    # print(unique_words)
    doc_word = []
    unique_words = sorted(unique_words)
    for docs in range(len(BoW)):
        doc_word.append(dict.fromkeys(unique_words, 0))
        for word in BoW[docs]:
            try:
                doc_word[docs][word] += 1
            except:
                continue
    tf_docs = []
    for i, document in enumerate(doc_word):
        tf_docs.append(tf_computation(document, BoW[i]))
    idf_dicts = idf_computation(doc_word)
    tf_idf_docs = []
    for tf_doc in tf_docs:
        tf_idf_docs.append(tf_idf_computation(tf_doc, idf_dicts))
    return tf_idf_docs


# importing dataset
training_set = pd.read_csv('trainset.csv')
validation_set = pd.read_csv('validationset.csv')
# diplaying first 10 rows of training set
# print(training_set.head(10))
training_set = training_set.head(1000)
validation_set = validation_set.head(500)
# droping unnecessory columns
training_set = training_set[['title', 'desc', 'cat1', 'cat2', 'cat3']]
validation_set = validation_set[['title', 'desc', 'cat1', 'cat2', 'cat3']]
'''# number of rows in each dataset
train_len = len(training_set.index)
valid_len = len(validation_set.index)
print(train_len)
print(valid_len)
print(training_set[['cat1', 'cat2', 'cat3']].count())'''
# preprocessing
X_train = training_set.iloc[:, :2]
Y_train = training_set.iloc[:, 2:5]
X_test = validation_set.iloc[:, :2]
Y_test = validation_set.iloc[:, 2:5]
# X_train['documents'] = X_train['title'].astype(str) + ' ' + X_train['desc'].astype(str)
clean_doc_train = []
for doc in X_train['title'].values:
    clean_doc_train.append(preprocessing(doc))
X_train['clean_documents'] = clean_doc_train
# X_test['documents'] = X_test['title'].astype(str) + ' ' + X_test['desc'].astype(str)
clean_doc_test = []
for doc in X_test['title'].values:
    clean_doc_test.append(preprocessing(doc))
X_test['clean_documents'] = clean_doc_test
# text vectorization
BoW_train = []
for row in clean_doc_train:
    BoW_train.append(row.split())
tf_idf_train = pd.DataFrame(tf_idf(clean_doc_train, BoW_train))
tf_idf_test = pd.DataFrame(tf_idf(clean_doc_test, BoW_train))
# one hot encoding
cat_list_train = []
for i in Y_train.index:
    cat_list_train.append(Y_train.iloc[i, :].values.tolist())

cat_list_test = []
for i in Y_test.index:
    cat_list_test.append(Y_test.iloc[i, :].values.tolist())

unique_tags = []
for cats in cat_list_train:
    unique_tags = set(unique_tags).union(set(cats))
for x in unique_tags:
    if x != x:
        unique_tags.remove(x)
        break
unique_tags = sorted(unique_tags)

doc_tags = []
for row in Y_train.index:
    doc_tags.append(dict.fromkeys(unique_tags, 0))
    for word in cat_list_train[row]:
        if word == word:
            doc_tags[row][word] = 1
doc_tags_test = []
for row in Y_test.index:
    doc_tags_test.append(dict.fromkeys(unique_tags, 0))
    for word in cat_list_test[row]:
        if word in unique_tags:
            doc_tags_test[row][word] = 1
Y_train = pd.DataFrame(doc_tags)
Y_test = pd.DataFrame(doc_tags_test)
# logistic regression
x = np.array(tf_idf_train)
y = np.array(Y_train)
x_t = np.array(tf_idf_test)
y_t = np.array(Y_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X.dot(theta))


def cost_function_reg(theta, X, y, reg_lambda):
    m = len(y)
    y_zero = (1 - y).dot(np.log(1 - h(X, theta)))
    y_one = y.dot(np.log(h(X, theta)))
    reg = (reg_lambda / (2 * m)) * sum(theta[1:] ** 2)
    J = (-1 / m) * (y_zero + y_one) + reg
    return J


def gradient_reg(theta, X, y, reg_lambda):
    m = len(y)
    reg = (reg_lambda / m) * theta
    reg[0] = 0
    return ((h(X, theta) - y).dot(X) / m) + reg


def one_vs_all(X, y, num_labels, reg_lambda=0.1):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))
    for c in range(1, num_labels + 1):
        initial_theta = np.zeros((n + 1, 1))
        theta = fmin_cg(f=cost_function_reg, x0=initial_theta, fprime=gradient_reg, args=(X, y == c, reg_lambda),
                        maxiter=100)
        all_theta[c - 1, :] = theta.T

    return all_theta


def predict_one_vs_all(all_theta, X):
    m = len(X)
    X = np.hstack((np.ones((m, 1)), X))
    return np.argmax(h(X, all_theta.T), axis=1) + 1


# final_theta = one_vs_all(x, y, 10, 0.1)

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
clf.fit(x, y)
y_pred = clf.predict(x_t)
y_pred = pd.DataFrame(y_pred)
y = pd.DataFrame(y)
print(y)
print(y_pred)
c_m = multilabel_confusion_matrix(y_t, y_pred)
accuracy = 0
sumT = 0
sumall = 0
for i in range(len(c_m)):
    sumT = c_m[i][1][1] + c_m[i][0][0]
    sumall = c_m[i][0][0] + c_m[i][1][1] + c_m[i][0][1] + c_m[i][1][0]
    accuracy += sumT/sumall

accuracy = accuracy/len(c_m)
print(accuracy)
precision = 0
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(len(c_m)):
    tp = c_m[i][1][1]
    fp = c_m[i][0][1]
    precision += tp/(tp + fp)
precision = precision/len(c_m)
print(precision)
recall = 0
for i in range(len(c_m)):
    fn = c_m[i][1][0]
    tp = c_m[i][1][1]
    recall += tp/(tp + fn)
recall = recall/len(c_m)
print(recall)
f1 = 2 * (recall * precision)/(recall + precision)
print(f1)
