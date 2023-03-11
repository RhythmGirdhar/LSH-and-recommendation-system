from pyspark import SparkContext
import sys
import time
import random
import operator
from itertools import combinations

def create_func_list():
    func_list = list()
    param_as = random.sample(range(1, m), n)
    func_list.append(param_as)
    param_bs = random.sample(range(1, m), n)
    func_list.append(param_bs)

    return func_list

def create_signature(func_list, business_user_dict, user_index_dict):
    sign_dict = dict()
    for business, users in business_user_dict.items():
        minhash_sign_list = list()
        for i in range(n):
            minhash = float("inf")
            for user in users:
                minhash = min(minhash, (((func_list[0][i] * user_index_dict[user] + func_list[1][i]) % p) % m))
            minhash_sign_list.append(int(minhash))
        sign_dict[business] = minhash_sign_list
    return sign_dict

def get_candidates(sign_dict):
    bands_dict = dict()
    for business, minhash_sign in sign_dict.items():
        for i in range(b):
            index = (i, tuple(minhash_sign[i * r: i * r + r]))
            if index not in bands_dict.keys():
                bands_dict[index] = []
            bands_dict[index].append(business)
    
    candidate_dict = {key: values for key, values in bands_dict.items() if len(values) > 1}

    candidate_pairs = set()
    for values in candidate_dict.values():
        sorted_values = sorted(values)
        comb_list = combinations(sorted_values, 2)
        for item in comb_list:
            candidate_pairs.add(item)
    
    sorted_pairs = sorted(candidate_pairs, key=lambda pair: (pair[0], pair[1]))
    return sorted_pairs

def check_similarity(candidate_pairs, business_user_dict):
    result_str = ""
    for bus1, bus2 in candidate_pairs:
        user1 = business_user_dict[bus1]
        user2 = business_user_dict[bus2]
        jaccard = len(user1 & user2) / len(user1 | user2)

        if jaccard >= 0.5:
            result_str += str(bus1) + "," + str(bus2) + "," + str(jaccard) + "\n"
    
    return result_str

def write_csv_file(result_str, output_file):
    with open(output_file, "w") as f:
        result_header = "business_id_1, business_id_2, similarity\n"
        f.writelines(result_header)
        f.writelines(result_str)

if __name__ == "__main__":
    sc = SparkContext()

    # input_file = sys.argv[1]
    # output_file = sys.argv[2]

    input_file = "../data/yelp_train.csv"
    output_file = "../result/task1.csv"
    ground_truth_file = "../data/pure_jaccard_similarity.csv"

    data_RDD = sc.textFile(input_file)
    header = data_RDD.first()
    data_RDD = data_RDD.filter(lambda row: row != header).map(lambda row: row.split(",")).cache()

    user_index_dict = data_RDD.map(lambda kv: kv[0]).distinct() \
            .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
            .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    business_user = data_RDD.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)

    business_user_dict = {}
    for business, users in business_user.collect():
        business_user_dict[business] = users

    
    #all variable declarations
    n = 60
    m = len(user_index_dict) * 2
    p = 233333333333
    r = 2
    b = n // r

    func_list = create_func_list()

    sign_dict = create_signature(func_list, business_user_dict, user_index_dict)

    candidate_pairs = get_candidates(sign_dict)
    
    result_str = check_similarity(candidate_pairs, business_user_dict)

    write_csv_file(result_str, output_file)


    """
    Calculate precision and recall
    """
    with open("../data/pure_jaccard_similarity.csv") as in_file:
        answer = in_file.read().splitlines(True)[1:]
    answer_set = set()
    for line in answer:
        row = line.split(',')
        answer_set.add((row[0], row[1]))
    with open("../result/task1.csv") as in_file:
        estimate = in_file.read().splitlines(True)[1:]
    estimate_set = set()
    for line in estimate:
        row = line.split(',')
        estimate_set.add((row[0], row[1]))
    print("Precision:")
    print(len(answer_set.intersection(estimate_set))/len(estimate_set))
    print("Recall:")
    print(len(answer_set.intersection(estimate_set))/len(answer_set))
    print(answer_set.difference(estimate_set))