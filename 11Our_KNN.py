import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')

# class k & r and their features.
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_feature = [5, 7]


# print data
# one line / https://youtu.be/n3RqsMz3-0A?t=300
[[plt.scatter(j[0], j[1], s=100, color=i) for j in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100, color='b')
plt.show()


def K_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is less than total voting groups')
    # KNN algorithm  remind: You can use radius, not calculate all data's distance.
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    print(votes)
    print(Counter(votes).most_common(1))
    # [('r', 3)] you can see that it have a tuple in list, so it should be [0][0].

    return vote_result

result = K_nearest_neighbors(dataset, new_feature, 3)

print(result)
