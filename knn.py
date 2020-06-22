import numpy as np
import random

def predict(data,k):
    train_count = int(len(data) * 0.8)
    test_count = int(len(data) - train_count)
    
    train_count=5
    test_count=5
    
    
    width = len(data[0]) - 1
    
    
    for i in range(test_count):
        nth = len(data)-1-i
        print(nth)
        
        distances = []
        for j in range(train_count):   
            tmp = 0
            for l in range(1,width+1):
                tmp = tmp + ( abs(float(data[j][l]) - float(data[nth][l]))**2 )         
            tmp = tmp**(1/width)
            
            distances.append(tmp)

        mins = []
        for j in range(k):
            mins.append(distances[j])
        mins.sort()
        print(mins)
        
        """for j in range(3,train_count):
            if"""
            
            
        
 
    










if __name__ == "__main__":
    k=3
    file=open('datasets/iris.data', 'r').readlines()
    N=len(file)
    
    data = []
    
    for i in range(0,N):
        line = file[i].split(",")
        line[-1] = line[-1].strip()
        data.append(line)

    #random.shuffle(data)
    predict(data,k)