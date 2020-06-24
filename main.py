from os import system
import numpy as np
import random

#split dataset into n folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

#calculates the mean accuracy of predictions
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

#calculates distance
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return distance**(1/2)

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
    
#finds the best match
def get_best_match(dataset, test_row):
	distances = list()
	for data in dataset:
		dist = euclidean_distance(data, test_row)
		distances.append((data, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

def predict(dataset, test_row):
	bm = get_best_match(dataset, test_row)
	return bm[-1]

def random_data(train):
	n_records = len(train)
	n_features = len(train[0])
	data = [train[random.randrange(n_records)][i] for i in range(n_features)]
	return data

def train_dataset(train, n_dataset, lrate, epochs):
	dataset = [random_data(train) for i in range(n_dataset)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		for row in train:
			bm = get_best_match(dataset, row)
			for i in range(len(row)-1):
				error = row[i] - bm[i]
				if bm[-1] == row[-1]:
					bm[i] += rate * error
				else:
					bm[i] -= rate * error
	return dataset

def evaluate_knn(data,k):
    train_count = int(len(data) * 0.8)
    test_count = int(len(data) - train_count)
    width = len(data[0]) - 1
    scores = np.zeros(2)
    for i in range(test_count):
        nth = len(data)-1-i
        distances = np.zeros((train_count,2))
        for j in range(train_count):   
            tmp = 0
            for l in range(1,width+1):
                tmp = tmp + ( abs(float(data[j][l]) - float(data[nth][l]))**2 )         
            tmp = tmp**(1/width)
            distances[j][1]=j
            distances[j][0]=tmp
        distances = distances[np.argsort(distances[:, 0])]
        temp=0
        for l in range(k):
            if( data[nth][0] == data[ int(distances[l][1]) ][0] ):
                temp = temp + 1
        if(temp>k/2):
            scores[0] = scores[0] + 1
        else:
            scores[1] = scores[1] + 1
    return scores

def lvq(train, test, n_dataset, lrate, epochs):
    dataset = train_dataset(train, n_dataset, lrate, epochs)
    predictions = list()
    for row in test:
        output = predict(dataset, row)
        predictions.append(output)
    return(predictions)

def evaluate_lvq(dataset, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = lvq(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
    
if __name__ == "__main__":
    system('cls')
    k=3
    n_folds = 5
    learn_rate = 0.3
    n_epochs = 50
    n_dataset = 20
    
    datasets = ["balance-scale", "iris", "abalone"]
    dataset_count=len(datasets)
    
    for j in range(dataset_count):
        print("%s dataset is calculating..." %datasets[j])
        filepath = "datasets/"
        filepath = filepath + datasets[j]
        filepath = filepath + ".data"
        
        file=open(filepath, 'r').readlines()
        N=len(file)
      
        data = []
        for i in range(0,N):
            line = file[i].split(",")
            line[-1] = line[-1].strip()
            data.append(line)
        random.shuffle(data)
       
        #start KNN algorithm
        scores = evaluate_knn(data,k)
        print('\tsuccess rate with KNN = %.2f' %(scores[0]/(scores[0]+scores[1])))
    
        for i in range(len(data)):
            tmp = data[i][0]
            data[i][0] = data[i][len(data[0])-1]
            data[i][len(data[0])-1] = tmp
            
        for i in range(len(data[0])-1):
            str_column_to_float(data, i) 
        str_column_to_int(data, len(data[0])-1)
        
        #start LVQ algorithm
        scores = evaluate_lvq(data, n_folds, n_dataset, learn_rate, n_epochs)
        print('\tsuccess rate with LVQ = %.2f' %(sum(scores)/float(len(scores)*100)),"\n")
        