import math
from pyspark import SparkContext
import sys
import time
import random
import operator
from itertools import combinations

def write_csv_file(result_str, output_file):
    with open(output_file, "w") as f:
        result_header = "user_id, business_id, prediction\n"
        f.writelines(result_header)
        f.writelines(result_str)

def calculateWeightedAverage(weight_list_candidates):
    X, Y = 0, 0
    for weight, rating in weight_list_candidates:
        X += weight * rating
        Y += abs(weight)
    return 3.5 if Y == 0 else X / Y

def computeSimilarity(co_rated_users, dict1, dict2):
    val1_list, val2_list = list(), list()

    [(val1_list.append(float(dict1[user_id])),
      val2_list.append(float(dict2[user_id]))) for user_id in co_rated_users]

    avg1 = sum(val1_list) / len(val1_list)
    avg2 = sum(val2_list) / len(val2_list)

    numerator = sum(map(lambda pair: (pair[0] - avg1) * (pair[1] - avg2), zip(val1_list, val2_list)))

    if numerator == 0:
        return 0
        
    denominator = math.sqrt(sum(map(lambda val: (val - avg1) ** 2, val1_list))) * \
                  math.sqrt(sum(map(lambda val: (val - avg2) ** 2, val2_list)))
    if denominator == 0:
        return 0

    return numerator / denominator

def item_based_cf(business, user):
    if (business not in business_user_dict) and (user not in user_business_dict):
        return DEFAULT_VALUE

    if business not in business_user_dict:
        return user_avg_rating[user]
    
    if user not in user_business_dict:
        return business_avg_rating[business]
    
    weight_list = list()

    for bus in user_business_dict[user]:
        business_bus = tuple(sorted((business, bus)))
        if business_bus in weight_dict:
            weight = weight_dict[business_bus]
        else:
            co_rated_users = list(set(business_user_dict[business]) & (set(business_user_dict[bus])))
            no_co_rated_users = len(co_rated_users)
            if no_co_rated_users <= 1:
                weight = (5.0 - abs(business_avg_rating[business] - business_avg_rating[bus])) / 5
            elif no_co_rated_users == 2:
                weight_1 = (5.0 - abs(float(bus_user_r_dict[business][co_rated_users[0]]) - float(bus_user_r_dict[bus][co_rated_users[0]]))) / 5
                weight_2 = (5.0 - abs(float(bus_user_r_dict[business][co_rated_users[1]]) - float(bus_user_r_dict[bus][co_rated_users[1]]))) / 5
                weight = (weight_1 + weight_2) / 2
            else:
                weight = computeSimilarity(co_rated_users, bus_user_r_dict[business], bus_user_r_dict[bus])
            
            weight_dict[business_bus] = weight
        
        if weight > 0:
            weight_list.append((weight, float(bus_user_r_dict[bus][user])))

    weight_list_candidates = sorted(weight_list, key = lambda x: -x[0])[:NEIGHBORS]

    prediction = calculateWeightedAverage(weight_list_candidates)

    return prediction


if __name__ == "__main__":
    sc = SparkContext()

    input_train_file = sys.argv[1]
    input_test_file = sys.argv[2]
    output_file = sys.argv[3]

    # input_train_file = "data/yelp_train.csv"
    # input_test_file = "data/yelp_val_in.csv"
    # output_file = "result/task2_1.csv"

    DEFAULT_VALUE = 3.5
    NEIGHBORS = 15

    train_data_RDD = sc.textFile(input_train_file)
    header = train_data_RDD.first()
    train_data_RDD = train_data_RDD.filter(lambda row: row != header).map(lambda row: row.split(","))

    test_data_RDD = sc.textFile(input_test_file)
    header = test_data_RDD.first()
    test_data_RDD = test_data_RDD.filter(lambda row: row != header).map(lambda row: row.split(","))

    business_user_dict = train_data_RDD.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).collectAsMap()
    user_business_dict = train_data_RDD.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).collectAsMap()
    business_avg_rating = train_data_RDD.filter(lambda row: row[2].replace('.', '', 1).isdigit()) \
                                         .map(lambda x: (x[0], float(x[2]))) \
                                         .groupByKey() \
                                         .mapValues(lambda x: sum(x)/len(x)) \
                                         .collectAsMap()

    user_avg_rating = train_data_RDD.filter(lambda row: row[2].replace('.', '', 1).isdigit()) \
                                         .map(lambda x: (x[1], float(x[2]))) \
                                         .groupByKey() \
                                         .mapValues(lambda x: sum(x)/len(x)) \
                                         .collectAsMap()

    bus_user_r = train_data_RDD.map(lambda row: (row[0], (row[1], row[2]))).groupByKey().mapValues(set).collect()

    bus_user_r_dict = dict()
    for bus, user_r_set in bus_user_r:
        temp = dict()
        for user_r in user_r_set:
            temp[user_r[0]] = user_r[1]
        bus_user_r_dict[bus] = temp
    
    weight_dict = dict()

    result_str = ""
    for row in test_data_RDD.collect():
        prediction = item_based_cf(row[0], row[1])
        result_str += row[1] + "," + row[0] + "," + str(prediction) + "\n"

    write_csv_file(result_str, output_file)



    with open("result/task2_1.csv") as in_file:
        guess = in_file.readlines()[1:]
    with open("data/yelp_val.csv") as in_file:
        ans = in_file.readlines()[1:]
    res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_guess = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_ans = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    large_small = {"large": 0, "small": 0}

    RMSE = 0
    for i in range(len(guess)):
        diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
        RMSE += diff**2
        if abs(diff) < 1:
            res["<1"] = res["<1"] + 1
        elif 2 > abs(diff) >= 1:
            res["1~2"] = res["1~2"] + 1
        elif 3 > abs(diff) >= 2:
            res["2~3"] = res["2~3"] + 1
        elif 4 > abs(diff) >= 3:
            res["3~4"] = res["3~4"] + 1
        else:
            res["4~5"] = res["4~5"] + 1
    RMSE = (RMSE/len(guess))**(1/2)
    print("RMSE: "+str(RMSE))