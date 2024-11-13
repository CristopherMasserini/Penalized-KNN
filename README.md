# Penalized-KNN
A Penalized KNN model

Ideas:
Essentially make a knn model where each neighbor is weighted base roof how many neighbors we are looking at. So if n=10, the closest has weight 1, second has weight 0.9, etc. So essentially weights =[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

If tie, whatever is seen first

Can have distance calculated euclidean or taxicab

need two files, one for model one for data

data should have subclass for each point (x1, x2, x3, …) and classifier (None or a string)
