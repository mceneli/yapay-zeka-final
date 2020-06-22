import numpy as np










if __name__ == "__main__":
    
    file=open('datasets/abalone.data', 'r').readlines()
    N=len(file)-1
    train_count = int(N * 0.8)
    test_count = int(N - train_count)
    
    for i in range(0,train_count):
        line = file[i].split(",")
        print(line)



    """filepath = "datasets/abalone.data"
    data = np.loadtxt(filepath)

    train_count = int(data.shape[0] * 0.8)
    test_count = int(data.shape[0] - train_count)

    print(train_count)
    print(test_count)

    traindata = np.zeros((train_count, 2))
    testdata = np.zeros((test_count, 2))

    for i in range(train_count):
        traindata[i][0]=data[i][0]
        traindata[i][1]=data[i][1]

    for i in range(train_count,data.shape[0]):
        testdata[i-train_count][0] = data[i][0]
        testdata[i-train_count][1] = data[i][1]
        

    print(traindata,"\n")
    print(testdata)"""