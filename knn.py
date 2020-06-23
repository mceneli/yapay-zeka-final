import numpy as np
import random

def predict(data,k):
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
        #print(data[nth][0])
        for l in range(k):
            #print(data[ int(distances[l][1]) ][0])
            if( data[nth][0] == data[ int(distances[l][1]) ][0] ):
                temp = temp + 1
        if(temp>k/2):
            scores[0] = scores[0] + 1
        else:
            scores[1] = scores[1] + 1
    return scores
            
if __name__ == "__main__":
    k=3
    datasets = ["abalone", "balance-scale", "iris"]
    dataset_count=len(datasets)
    
    for j in range(dataset_count):
        print("calculated dataset = ",datasets[j])
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
        scores = predict(data,k)
        print("success rate = ",scores[0]/(scores[0]+scores[1]),"\n")
    
    