import pandas as pd
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Calculates the number of characters in the reviews
def calcLength(texts):
    lengths = []
    for i in range(len(texts)):
        if type(texts[i]) == float:
            lengths.append(0)
            continue
        lengthh = len(texts[i])
        lengths.append(lengthh)
        
    new_data["length"] = lengths
    new_data.to_csv("inputData.csv", index=False)
    print('Done calculating lengths of the reviews')

# Calculates the total number of '?' and '!' in the reviews
def calcEmotion(texts):
    punctuations = []
    for text in texts:
        if type(text) == float:
            punctuations.append(0)
            continue
        exclamation = [1 if '!' in sample else 0 for sample in text]
        questionMark = [1 if '?' in sample else 0 for sample in text]
        numberOfex = 0
        numberOfqs = 0
        for i in exclamation:
            if i == 1:
                numberOfex += 1
        for i in questionMark:
            if i == 1:
                numberOfqs += 1
        sum = numberOfqs + numberOfex
        punctuations.append(sum)
        
    new_data["punctuations"] = punctuations
    new_data.to_csv("inputData.csv", index=False)
    print('Done calculating punctuation marks of the reviews')

# Calculates the ARI of the reviews
def calcAri(texts):
    punc_sent = string.punctuation
    upper = string.ascii_uppercase
    text_length = []
    num_sentence = []
    readibility = []

    for i in texts:
        if type(i) == float:
          readibility.append(0)
          continue
        texti = i
        text_lengthi = 0
        num_sentencei = 0
        num_wordsi = len(texti.split(" "))
        for j in range(len(texti)):
            if texti[j].isalpha():
                text_lengthi += 1
            if j <= len(texti) - 3:
                if texti[j] in punc_sent and (texti[j+1] in upper or texti[j+2] in upper):
                    num_sentencei += 1
        if num_wordsi == 0:
            num_wordsi = 1
        if num_sentencei == 0:
            num_sentencei = 1
        readibilityi = 4.71 * (text_lengthi/num_wordsi) + 0.5 * (num_wordsi/num_sentencei) - 21.34
        text_length.append(text_lengthi)
        num_sentence.append(num_sentencei)
        readibility.append(readibilityi)

    new_data["ari"] = readibility
    new_data.to_csv("inputData.csv", index=False)
    print('Done calculating ARI of the reviews')


# Calculates the percentage of people who found the review helpful
def helpfulReviews(reviews):
    rating = []
    ratingi = 0
    for i in range(len(reviews)):
        review = reviews[i]
        if type(review)==float:
          rating.append(0)
          continue
        raw = review.replace("[", "")
        raw2 = raw.replace("]", "")
        raw3 = raw2.split(", ")
        num1 = float(raw3[0])
        num2 = float(raw3[1])
        if num1 == 0 or num2 == 0:
          ratingi = 0
        else:
          ratingi = num1/num2
        rating.append(ratingi)
    new_data["helpfulreviewsvalue"] = rating
    new_data.to_csv("inputData.csv", index=False)
    print('Done calculating helpfulness rating of the reviews')


# Marks the reviews helpful or not helpful based on percentage
def decideValues(ratings):
    vals = []
    for i in range(len(ratings)):
        result = 0
        rating = ratings[i]
        # change this 0.75 if you want
        if rating > 0.75:
            result = 1
        vals.append(result)
    new_data["helpfulornot"] = vals
    new_data.to_csv("inputData.csv", index=False)
    print('Done calculating the preprocessed values if the reviews were helpful or not')

# Data preprocessing start
# import file and drop all NaNs
data = pd.read_csv("inputData.csv", engine='python', error_bad_lines=False)
new_data = data.dropna(axis = 1, how ='all') 

features = ['reviewText', 'helpful']
features_process = pd.get_dummies(new_data[features])

print("Starting model...")
print("Starting data pre-processing...")

calcLength(new_data['reviewText'])
calcEmotion(new_data['reviewText'])
helpfulReviews(new_data['helpful'])
calcAri(new_data['reviewText'])
decideValues(new_data['helpfulreviewsvalue'])

data = pd.read_csv("inputData.csv",engine='python', error_bad_lines=False)
new_data = data.dropna(axis = 1, how ='all')

print("Ending data pre-processing...")

# Data preprocessing end

features = ['ari','helpfulreviewsvalue','overall','length','punctuations']
#features = ['ari','helpfulreviewsvalue','overall']

y = data['helpfulornot']
X = pd.get_dummies(data[features])

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

print("Starting data training...")

# Trainning our model
clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf2.fit(X_train, y_train)
classifier1 = KNeighborsClassifier(n_neighbors=6, algorithm='brute')
classifier1.fit(X_train, y_train)  
clf3 = GaussianNB()

print("Ending data training...")
print("Making data predictions for 20% of the data...")

# Predicitions
pred3 = clf3.fit(X_train, y_train).predict(X_test)
pred2 = clf2.predict(X_test)
pred1 = classifier1.predict(X_test)

print("The accuracy score for SVM is:")
accuracy_score(y_test, pred2)

# KNN => pred1 
# SVM => pred2
# GNB => pred3
