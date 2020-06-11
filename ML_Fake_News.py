#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import Counter
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import csv
import seaborn as sns


# In[2]:


def parseJson(fname):
    for line in open(fname, 'r'):
        yield eval(line)


# In[3]:


data = list(parseJson('E:/IRELAND/MATERIALS/EE514/Assignment/fake_news.json'))


# In[6]:


df = pd.DataFrame(data)
csv_data = df.to_csv('E:/IRELAND/MATERIALS/EE514/Assignment/NEW_DATA/fake_news.csv',index=False)


# In[9]:


df = pd.read_csv('E:/IRELAND/MATERIALS/EE514/Assignment/NEW_DATA/fake_news.csv')
df_v2 = pd.read_csv('E:/IRELAND/MATERIALS/EE514/Assignment/NEW_DATA/fake_news.csv')


# In[11]:


print(df.shape)
print(df_v2.shape)


# In[12]:


print(df.isnull().values.any())
print(df_v2.isnull().values.any())


# In[13]:


df.head()


# In[14]:


df1 = pd.Series(' '.join(df.headline[df.is_sarcastic==1]).lower().split()).value_counts()
df1=df1.head(20)
print (df1)
df1.columns=["Words","Frequency"]
df1.plot(x='Words',y='Frequency',title= 'TOP 20 WORDS IN FAKE NEWS',kind='bar',color='yellow')


# In[28]:


df2 = pd.Series(' '.join(df.headline[df.is_sarcastic==0]).lower().split()).value_counts()
df2=df2.head(20)
print (df2)
df2.columns=["Words","Frequency"]
df2.plot(x='Words',y='Frequency',title= 'TOP 20 WORDS IN REAL NEWS',kind='bar',color='purple')


# In[15]:


import nltk
nltk.download('stopwords')


# In[24]:


print(df_v2.shape)
print(df.shape)
print(df.is_sarcastic.shape)


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
fake_text = df_v2.headline[df_v2.is_sarcastic==1].str.split()
print("\n Original Fake News Headline: \n")
print(df.headline[df.is_sarcastic==1].head(20))
eliminate_fake_text = fake_text.apply(lambda text: [item for item in text if item not in stoplist])
print("\n After removing stop words from the Fake News Headline: \n")
eliminated_fake_text = eliminate_fake_text.str.join(' ')
print(eliminated_fake_text.head(20))

# Set variables to show Fake Headlines Titles
mask = df_v2['is_sarcastic'] == 1
df_fake_headlines = df[mask]['headline']

# Instantiate a CountVectorizer
cv = CountVectorizer(stop_words = 'english', max_df=0.7)

# Fit and transform the vectorizer
fake_cvec = cv.fit_transform(df_fake_headlines)

# Convert fake_cvec into a DataFrame
fake_cvec_df = pd.DataFrame(fake_cvec.toarray(),
                   columns=cv.get_feature_names())

# Inspect head of Fake Headline Titles cvec
print(fake_cvec_df.shape)

def bar_plot(x, y, title, color):    
    
    # Set up barplot 
    plt.figure(figsize=(9,5))
    g=sns.barplot(x, y, color = color)    
    ax=g

    # Label the graph
    plt.title(title, fontsize = 15)
    plt.xticks(fontsize = 10)

    # Enable bar values
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for p in ax.patches:
        totals.append(p.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+.3, p.get_y()+.38,                 int(p.get_width()), fontsize=10)


# In[26]:


# Set up variables to contain top 10 most used words in Fake News Headlines
fake_wc = fake_cvec_df.sum(axis = 0)
fake_top_5 = fake_wc.sort_values(ascending=False).head(10)

# Call function
bar_plot(fake_top_5.values, fake_top_5.index,'Frequently used words in Fake News Headlines','r')


# In[27]:


real_text = df_v2.headline[df_v2.is_sarcastic==0].str.split()
print("\n Original Real News Headline: \n")
print(df.headline[df.is_sarcastic==0].head(20))
eliminate_real_text = real_text.apply(lambda text: [item for item in text if item not in stoplist])
print("\n After removing stop words from the Real News Headline: \n")
eliminated_real_text = eliminate_real_text.str.join(' ')
print(eliminated_real_text.head(20))
# Set up variables to contain top 5 most used bigrams in r/TheOnion
real_wc = real_cvec_df.sum(axis = 0)
real_top_5 = real_wc.sort_values(ascending=False).head(10)

# Call function
bar_plot(real_top_5.values, real_top_5.index,'Frequently used words in Real News Headlines','blue')


# In[29]:


X_split_data=df.headline.str.replace('\d+', '')
print(X_split_data)
Y_split_data=df.is_sarcastic
print(Y_split_data)


# In[30]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)


# In[32]:


# Initialize the `tfidf_vectorizer` 
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set 
tfidf_test = tfidf_vectorizer.transform(X_test)


# In[33]:


# Get the feature names of `tfidf_vectorizer` 
print(tfidf_vectorizer.get_feature_names())


# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix:")

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[37]:


N = 8
comp1 = pd.Series(df.headline[df.is_sarcastic==0])
real_headline = comp1.str.len()
real_headline1 = real_headline.head(8)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
heads1 = ax.bar(ind, real_headline1, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Length of headlines')
ax.set_title('Real News Headlines')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'))

#ax.legend((rects1[0], ('Real Headline')))
def headlinelabel(heads):
    
    for hd in heads:
        height = hd.get_height()
        ax.text(hd.get_x() + hd.get_width()/2., 1.005*height,
                '%d' % int(height),
                ha='center', va='bottom')

headlinelabel(heads1)
plt.show()


# In[39]:


N = 8
comp1 = pd.Series(df.headline[df.is_sarcastic==1])
fake_headline = comp1.str.len()
fake_headline1 = fake_headline.head(8)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
heads1 = ax.bar(ind, fake_headline1, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('Length of headlines')
ax.set_title('Fake News Headlines')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'))

def headlinelabel(heads):
    
    for hd in heads:
        height = hd.get_height()
        ax.text(hd.get_x() + hd.get_width()/2., 1.005*height,
                '%d' % int(height),
                ha='center', va='bottom')

headlinelabel(heads1)
plt.show()


# In[40]:


N = 8
comp1 = pd.Series(df.headline[df.is_sarcastic==0])
real_headline = comp1.str.len()
real_headline1 = real_headline.head(8)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
heads1 = ax.bar(ind, real_headline1, width, color='r')

comp2 = pd.Series(df.headline[df.is_sarcastic==1])
fake_headline = comp2.str.len()
fake_headline1 = fake_headline.head(8)
heads2 = ax.bar(ind + width, fake_headline1, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Length of headlines')
ax.set_title('Real and Fake Headlines')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'))

ax.legend((heads1[0], heads2[0]), ('Real Headline', 'Fake Headline'))


def headlinelabel(heads):
    
    for hd in heads:
        height = hd.get_height()
        ax.text(hd.get_x() + hd.get_width()/2., 1.005*height,
                '%d' % int(height),
                ha='center', va='bottom')

headlinelabel(heads1)
headlinelabel(heads2)

plt.show()


# In[74]:


import matplotlib as mpl
import matplotlib.pyplot as plt
data_to_plot = [fake_headline,real_headline]
fig = plt.figure(1, figsize=(6, 6))
ax = fig.add_subplot(111)
ax.set_ylabel('Length of headlines')
ax.set_xticklabels(['Fake News Headline', 'Real News Headline'])
bp = ax.boxplot(data_to_plot,patch_artist=True)


# In[41]:


import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


# In[42]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
classifier3 = DecisionTreeClassifier(criterion="entropy", max_depth=7)
classifier3 = classifier3.fit(tfidf_train, y_train)
y_pred = classifier3.predict(tfidf_test)
score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy obtained for testing set of Decision Tree Classifier: %0.3f" % score)
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
classifier3a = DecisionTreeClassifier(criterion="entropy", max_depth=7)
classifier3a = classifier3a.fit(tfidf_train, y_train)
y_pred = classifier3a.predict(tfidf_train)
score = metrics.accuracy_score(y_train, y_pred)
print("Accuracy obtained for training set of Decision Tree Classifier: %0.3f" % score)
cm = metrics.confusion_matrix(y_train, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[96]:


filename = 'E:/IRELAND/MATERIALS/EE514/Assignment/EE514_DecisionTree.sav'
pickle.dump(model, open(filename, 'wb'))


# In[43]:


from sklearn import svm
clf4 = svm.SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
clf4 = clf4.fit(tfidf_train, y_train)
y_pred = clf4.predict(tfidf_test)
score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy obtained for testing set of SVM: %0.3f" % score)
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[87]:


clf4 = svm.SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
clf4 = clf4.fit(tfidf_train, y_train)
y_pred = clf4.predict(tfidf_train)
score = metrics.accuracy_score(y_train, y_pred)
print("Accuracy obtained for training set of SVM: %0.3f" % score)
cm = metrics.confusion_matrix(y_train, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[97]:


filename = 'E:/IRELAND/MATERIALS/EE514/Assignment/EE514_SVM.sav'
pickle.dump(model, open(filename, 'wb'))


# In[44]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
logreg = logreg.fit(tfidf_train,y_train)
y_pred=logreg.predict(tfidf_test)
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision_score = metrics.precision_score(y_test, y_pred)
recall_score = metrics.recall_score(y_test, y_pred)
print("Accuracy for testing set of Logistic Regression: %0.3f" % accuracy_score)
print("Precision for testing set of Logistic Regression: %0.3f" % precision_score)
print("Recall for testing set of Logistic Regression: %0.3f" % recall_score)
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[86]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
logreg = logreg.fit(tfidf_train,y_train)
y_pred=logreg.predict(tfidf_train)
accuracy_score = metrics.accuracy_score(y_train, y_pred)
precision_score = metrics.precision_score(y_train, y_pred)
recall_score = metrics.recall_score(y_train, y_pred)
print("Accuracy for training set of Logistic Regression: %0.3f" % accuracy_score)
print("Precision for training set of Logistic Regression: %0.3f" % precision_score)
print("Recall for training set of Logistic Regression: %0.3f" % recall_score)
cm = metrics.confusion_matrix(y_train, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[98]:


filename = 'E:/IRELAND/MATERIALS/EE514/Assignment/EE514_LogisticRegression.sav'
pickle.dump(model, open(filename, 'wb'))


# In[46]:


from sklearn.naive_bayes import MultinomialNB
classifier1 = MultinomialNB()
import sklearn.metrics as metrics
classifier1.fit(tfidf_train, y_train)
pred = classifier1.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy obtained for testing set of MultinomialNB: %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[89]:


from sklearn.naive_bayes import MultinomialNB
classifier1 = MultinomialNB()
import sklearn.metrics as metrics
classifier1.fit(tfidf_train, y_train)
pred = classifier1.predict(tfidf_train)
score = metrics.accuracy_score(y_train, pred)
print("Accuracy obtained for training set of MultinomialNB: %0.3f" % score)
cm = metrics.confusion_matrix(y_train, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)


# In[48]:


y_train.shape


# In[49]:


#Splitting the training set into validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
#X_train, X_val, y_train, y_val = train_test_split(X_split_data, y_split_data, test_size=0.25, random_state=0)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_val = tfidf_vectorizer.transform(X_val)


# In[50]:


from sklearn.naive_bayes import MultinomialNB
classifier1 = MultinomialNB()
import sklearn.metrics as metrics
classifier1.fit(tfidf_train, y_train)
pred = classifier1.predict(tfidf_val)
score = metrics.accuracy_score(y_val, pred)
print("Accuracy obtained for validation set of MultinomialNB: %0.3f" % score)


# In[52]:


print(tfidf_train.shape)
print(y_train.shape)


# In[53]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=40)
kf.get_n_splits(X_split_data)
print(kf)
kf_model= MultinomialNB() 
accuracy_kf=cross_val_score(kf_model,tfidf_train , y_train, cv= 20)
print("Accuracy: %0.3f" % accuracy_kf.mean(axis=0))


# In[54]:


X_split_data.shape


# In[55]:


Y_split_data.shape


# In[56]:


tfidf_train.shape


# In[57]:


y_train.shape


# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
model=MultinomialNB()
model.fit(tfidf_train, y_train)
probs = model.predict_proba(tfidf_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[101]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_split_data, Y_split_data, test_size=0.25, random_state=0)
from pandas import DataFrame
export_csv = X_train.to_csv (r'E:/IRELAND/MATERIALS/EE514/Assignment/x_data_train.csv', index = None, header=True)
export_csv = X_test.to_csv (r'E:/IRELAND/MATERIALS/EE514/Assignment/x_data_test.csv', index = None, header=True)
export_csv = y_test.to_csv (r'E:/IRELAND/MATERIALS/EE514/Assignment/y_data_test.csv', index = None, header=True)
export_csv = y_train.to_csv (r'E:/IRELAND/MATERIALS/EE514/Assignment/y_data_train.csv', index = None, header=True)


# In[102]:


filename = 'E:/IRELAND/MATERIALS/EE514/Assignment/EE514_MultinomialNB.sav'
pickle.dump(model, open(filename, 'wb'))


# In[104]:


filename = 'E:/IRELAND/MATERIALS/EE514/Assignment/EE514_final_model.sav'
pickle.dump(model, open(filename, 'wb'))

