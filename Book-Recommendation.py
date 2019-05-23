import pandas as pd
import numpy as np
import math


def pearsoncorrelation(object1,traindata):
    object1 = object1.clip(0)

    allvalues = dict()

    for key,object2 in traindata.items():
        object2 = object2.clip(0)
        result1 = np.corrcoef(object1,object2)[0,1]
        if(np.isfinite(result1)):
            allvalues[result1]=key


    mylist = list(allvalues.keys())
    myother = sorted(mylist,reverse=True)
    knn = dict()
    neighbours = list()
    k = 0
    for i in allvalues:
        knn[allvalues[myother[k]]] = myother[k]
        neighbours.append(allvalues[myother[k]])
        k+=1

    return neighbours


def cosinesimilarity(a,traindata):

    a = a.clip(0)
    similarity_key = dict()
    norm_a = np.linalg.norm(a)
    for key,b in traindata.items():
        b = b.clip(0)
        dot_product = np.dot(a, b)
        norm_b = np.linalg.norm(b)
        result1= dot_product / (norm_a * norm_b)
        if(result1 != 0):
            similarity_key[key]=result1

    sorted_neigbours = sorted(similarity_key.items(), key=lambda x: x[1])
    neighbours = list()
    reversed(sorted_neigbours)

    for i,v in sorted_neigbours:
        if(np.isfinite(v)):
            print(i,v)
            neighbours.append(i)
    neighbours = reversed(neighbours)
    return neighbours


def cosinesimilarity1(a,traindata):

    a = a.clip(0)
    norm_a = np.norm(a)

    for key,b in traindata.items():
        b = b.clip(0)
        dot_product = np.dot(a, b)
        norm_b = np.norm(b)
        result1 = dot_product / (norm_a * norm_b)

    return result1


def adjustedCosineMatrix(a,traindata):

    users = traindata.index.size
    books = traindata.colums.size

    similar = np.zeros((users, books))
    avarage = np.array(users)

    for i in range(books):
        for j in range(i, book):
            similar[i][j] = cosinesimilarity1(traindata[:i] - avarage,traindata[:j] - avarage)
    return similar

def findusersratebook(all_rates,user_matrix,location):
    users_rated_book_indexs = list()

    count = -1
    for x in all_rates:
        count += 1
        if (x[location] != -1):
            users_rated_book_indexs.append(count)

    dict_index = {}

    for i in range(len(users_rated_book_indexs)):
        dict_index[user_matrix[users_rated_book_indexs[i]]] = all_rates[users_rated_book_indexs[i]]

    return  dict_index


def MAE(real_values, predicted_values):
    return np.abs(real_values - predicted_values).mean()

def weightedknn(location,user,neigbours,user_and_rates_matrix):
    weighted_rating_predict = 0
    a = len(user)
    user = user.clip(0)
    distance = list()

    m = 0
    for x in neigbours:
        if(m<3):
            m+=1
            y = user_and_rates_matrix.get(x)
            y = y.clip(0)
            suma = sum([abs(user[i]-y[i]) for i in range(a)])
            distance.append(suma)



    k_number = 0
    total_distance = 0

    for i in neigbours:
        if (k_number < 3):
            weighted_rating_predict = weighted_rating_predict + (((1/distance[k_number])) * user_and_rates_matrix.get(i)[location])
            total_distance = total_distance + (((1/distance[k_number])))
            k_number += 3


    return weighted_rating_predict/total_distance


def manhattan_distance(first, second):

    sum = 0
    for i in range(len(first)):
        x = abs(first[i]-second[i])
        sum+= x
    return sum

def knn(user_and_rates_matrix,neighbourlist,location):

    rating_predictionknn = 0
    k_number = 0

    for i in neighbourlist:
        if(k_number<3):
            k_number+=1
            rating_predictionknn= rating_predictionknn + user_and_rates_matrix.get(i)[location]


    return rating_predictionknn/3





book = pd.read_csv("BX-Books.csv",sep=";",error_bad_lines=False,encoding='latin-1')
book.columns = ["ISBN","Book-Title","Book-Author","Year-Of-Publication","Publisher","Image-URL-S","Image-URL-M","Image-URL-L"]

del book["Image-URL-S"]
del book["Image-URL-M"]
del book["Image-URL-L"]



user = pd.read_csv("BX-Users.csv",sep=";",error_bad_lines=False,encoding='latin-1')
user.columns = ("UserID","Location","Age")
user = user[user['Location'].str.contains("usa|canada")]


ratingforfind = pd.read_csv("train-sklearn.csv",sep=";",error_bad_lines=False,encoding='latin-1')
ratingforfind.columns = ("indexes","UserID","ISBN","BookRating")


# counts = ratingforfind["UserID"].value_counts()
# ratingforfind = ratingforfind[ratingforfind["UserID"].isin(counts[counts>=5].index)]
# counts1 = ratingforfind["BookRating"].value_counts()
# ratingforfind = ratingforfind[ratingforfind["BookRating"].isin(counts1[counts1>=5].index)]


rating = pd.merge(book,ratingforfind,on="ISBN")
rating = pd.merge(rating,user,on="UserID")
rating = rating[rating.BookRating != 0]




ratingpivot = rating.pivot_table(index='UserID', columns='ISBN', values='BookRating',fill_value=-1)



rating_pivot_all_rates = ratingpivot.values
rating_pivot_user_matrix = ratingpivot.index.values
rating_pivot_books_matrix = ratingpivot.columns.values
user_and_rates_matrix= dict(zip(rating_pivot_user_matrix, rating_pivot_all_rates))




test = pd.read_csv("test-sklearn.csv",sep=";",error_bad_lines=False,encoding='latin-1')
test.columns = ("indexes","UserID","ISBN","BookRating")



test = pd.merge(book,test,on="ISBN")
test = pd.merge(test,user,on="UserID")
test = test[test.BookRating != 0]

test_userID = test.UserID.values
test_ISBN = test.ISBN.values
test_ratings = test.BookRating.values

user_and_realrates_matrix= dict(zip(test_userID,test_ratings))

prediction_ratings_knn = []
prediction_ratings_wknn = []
im = -1
weighted_rating_prediction = 0

for isbn,user in zip(test_ISBN,test_userID):
    try:
        dict_index = findusersratebook(rating_pivot_all_rates,rating_pivot_user_matrix,ratingpivot.columns.get_loc(isbn))
        neighbourlist = pearsoncorrelation(rating_pivot_all_rates[ratingpivot.index.get_loc(user)],dict_index)
        im+=1
        rating_prediction = knn(user_and_rates_matrix,neighbourlist,ratingpivot.columns.get_loc(isbn))
        prediction_ratings_knn = np.append(prediction_ratings_knn,round(rating_prediction))
        weighted_rating_prediction = weightedknn(ratingpivot.columns.get_loc(isbn),rating_pivot_all_rates[ratingpivot.index.get_loc(user)],neighbourlist,user_and_rates_matrix)
        prediction_ratings_wknn = np.append(prediction_ratings_wknn,round(weighted_rating_prediction))
        print("knn : ",rating_prediction,"weight : ",weighted_rating_prediction )
    except KeyError:
        test_ratings = np.delete(test_ratings,im)




print((MAE(test_ratings, prediction_ratings_knn)))

print((MAE(test_ratings, weighted_rating_prediction)))