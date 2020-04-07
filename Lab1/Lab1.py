import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from torchvision import transforms
import time
import sys
import pickle
from PIL import Image
import math
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def process_bar(percent, start_str='', end_str='', total_length=0):
    # print the process bar
    bar = ''.join(["\033[31m%s\033[0m"%'-'] * int(percent * total_length)) + '>'
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)

def Q1():
    a_list = np.random.randint(10,size=10)
    print('The Question 1')
    print('The list is: ', a_list)
    print('The sum of the list is:', sum(a_list))
    return 0

def Q2():
    a_list = np.random.randint(10,size=10)
    print('The Question 2')
    print('The list is: ', a_list)
    print('The new list is:', list(set(a_list)))
    return 0

def Q3():
    print('The Question 3')
    string = input('Please input a string: ')
    string = string.replace(' ','')
    if string[::] == string[::-1]:
        print('It\'s a palindrome.')
    else:
        print('It\'s not a palindrome.')

def Q4():
    print('The Question 4')
    x = np.array([1+0j, 0.707+0.707j])
    print(x)
    solu = np.empty((x.shape[0],2))
    for i in range(x.shape[0]):
        solu[i] = [np.real(x[i]), np.imag(x[i])]
    print(solu)

def Q5():
    print('The Question 5')
    # Align the binary digits
    a = input('Please input 2 binary number: ').split(',')
    maxl = max(len(a[0]), len(a[1]))
    for i in range(2):
        a[i] = '0'*(maxl - len(a[i])) + a[i]
    solution = ''
    enter = 0  # Carry digits
    for i in range(maxl-1, -1, -1):
        tot = enter + int(a[0][i]) + int(a[1][i])
        if tot == 0:
            solution = '0' + solution
        elif tot == 1:
            solution = '1' + solution
        elif tot == 2:
            solution = '0' + solution
            enter = 1
        elif tot == 3:
            solution = '1' + solution
            enter = 1
    if enter == 1:
        solution = '1' + solution
    print('The solution is ', solution)
        
def Q6():
    print('The Question 6')
    # Define the ListNode class
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None

    def linkstore(link):
        # Store the link
        l = link.split('->')
        head = ListNode(int(l[0]))
        point = head
        for i in range(1,len(l)):
            new = ListNode(int(l[i]))
            point.next = new
            point = point.next
        return head
    
    def formula2link(formula):
        # Transfer the formula to link node
        formula = formula.split('+')
        assert len(formula) == 2, 'Not a right formula'
        head1 = linkstore(formula[0].rstrip(')').lstrip('('))
        head2 = linkstore(formula[1].rstrip(')').lstrip('('))
        return head1, head2
    
    def callinkplus(head1, head2):
        # plus the two link
        solution = ListNode(0)
        p_solu = solution
        carry_bit = 0
        while head1 or head2:
            x1 = head1 if head1 else 0
            x2 = head2 if head2 else 0
            solu = x1.val + x2.val + carry_bit
            carry_bit = solu // 10  # the plus calculation
            new = ListNode(solu%10)
            p_solu.next = new
            p_solu = p_solu.next
            head1 = head1.next
            head2 = head2.next
        return solution
    
    def printlink(link):
        # print the link
        point = link.next
        while point.next:
            print(point.val, end = '->')
            point = point.next
        print(point.val)
        return 0
          

    formula = input('Please input a formula: ')
    head1, head2 = formula2link(formula)
    solu1 = callinkplus(head1, head2)
    printlink(solu1)

def Q7():
    print('The Question 7')
    # Initialize
    a_list = np.random.randint(10,size=10)
    def bubblesort(sort_list):
        if len(sort_list) < 2:
            return sort_list
        for i in range(len(sort_list)):
            for j in range(i+1, len(sort_list)):
                if sort_list[i] > sort_list[j]:
                    temp = sort_list[i]
                    sort_list[i] = sort_list[j]
                    sort_list[j] = temp

        return sort_list
    print('The raw data is: ', a_list)
    print('The solution is: ', bubblesort(a_list))

def Q8():
    print('The Question 8')
    a_list = [random.randint(1, 50) for i in range(10)]
    print('The raw data is: ', a_list)
    def merge(left, right):
        solu = []
        while left and right:
            if left[0] < right[0]:
                solu.append(left.pop(0))
            else:
                solu.append(right.pop(0))
        if left:
            solu += left
        if right:
            solu += right
        return solu
    
    def mergesort(sort_list):
        if len(sort_list) < 2:
            return sort_list
        mid = len(sort_list)//2
        left = sort_list[:mid:]
        right = sort_list[mid::]
        return merge(mergesort(left),mergesort(right))
        
    print('The solution is: ', mergesort(a_list))

def Q9():
    print('The Question 9')
    a_list = np.random.randint(10,size=10)

    def quicksort(sort_list):
        if len(sort_list) < 2:
            return sort_list
        index = sort_list[0]
        left = []
        right = []
        for i in range(1,len(sort_list)):
            if sort_list[i] < index:
                left.append(sort_list[i])
            else: 
                right.append(sort_list[i])
        return quicksort(left) + [index] + quicksort(right)
    print('The raw data is: ', a_list)
    print('The solution is: ', quicksort(a_list))

def Q10():
    print('The Question 10')
    a_list = np.random.randint(10,size=20)

    def shellsort(sort_list):
        length = len(sort_list)
        gap = length // 2
        while gap > 0:
            for i in range(gap, length):
                for j in range(i, 0, -gap):
                    if sort_list[j] < sort_list[j-gap]:
                        temp = sort_list[j]
                        sort_list[j] = sort_list[j-gap]
                        sort_list[j-gap] = temp
            gap //= 2
        return sort_list                   
    print('The raw data is: ', a_list)
    print('The solution is: ', shellsort(a_list))

def Q11():
    print('The Question 11')
    # Using a linear layer to calculate the linearregression
    class LinearRegression(nn.Module):
        def __init__(self, input_size = 1, output_size = 1):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            return self.linear(x)
    # Initialize the data
    x = [3, 4, 5, 6, 7, 8, 9]
    y = []
    xx = np.linspace(2,10)
    x_d = np.array(x)
    for i in x:
        y.append(i * 2 + random.normalvariate(0,1))
    
    iteration = 500
    x = torch.FloatTensor(x)
    x = x.view(-1,1)
    x = Variable(x, requires_grad = True)
   
    xx = torch.FloatTensor(xx)
    xx = xx.view(-1,1)
    xx = Variable(xx)

    y = torch.FloatTensor(y)
    y = y.view(-1,1)
    y = Variable(y)

    #Set the parameter of the model
    input_size = 1
    output_size = 1
    model = LinearRegression(input_size, output_size)
    # The Loss function
    MSELoss = nn.MSELoss()

    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # Do the training
    for i in range(iteration):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = MSELoss(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch {}, Loss = {}'.format(i, loss.data))

    # Predict and plot
    perdict = model(xx).data.numpy()
    plt.scatter(x_d, y.detach().numpy(), color='r')
    plt.plot(xx, perdict, 'k-')
    plt.show()

def Q12():
    print('The Question 12')
    # We use a linear layer and a sigmoid layer to generate the regression model.
    class LogsticRegression(nn.Module):
        def __init__(self, input_size = 1, output_size = 1):
            super(LogsticRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.linear(x))
    iteration = 500
    # Set the data
    xx = np.linspace(1,4)

    x = np.linspace(2,8) 
    y = [[0] * 25 + [1] * 25]
    xx = np.linspace(2,10)
    x_d = np.array(x)
    
    iteration = 500
    x = torch.FloatTensor(x)
    x = x.view(-1,1)
    x = Variable(x, requires_grad = True)
   
    xx = torch.FloatTensor(xx)
    xx = xx.view(-1,1)
    xx = Variable(xx)

    y = torch.FloatTensor(y)
    y = y.view(-1,1)
    y = Variable(y)

    Model = LogsticRegression()
    BCELoss = nn.BCELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr = 1e-2)

    # Fit the data
    for i in range(iteration):
        y_pred = Model(x)
        loss = BCELoss(y_pred, y)
        if i % 100 == 0:
            print('Epoch {}, Loss = {}'.format(i, loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    xx = torch.FloatTensor(xx)
    xx = xx.view(-1,1)
    xx = Variable(xx)
    # Predict and plot
    perdict = Model(xx).data.numpy()
    xx = np.linspace(2,10)
    plt.scatter(x_d, y.detach().numpy(), color='b')
    plt.plot(xx, perdict, 'k-')
    plt.show()  

def Q13_14():
    print('The Question 13/14')
    # Set the seed of random number
    random.seed(0)
    # Set up the SVM model
    class SVM(nn.Module):
        def __init__(self, input_size = 2, output_size = 1):
            super(SVM, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            return self.linear(x)
    iteration = 500
    # Initialize data
    x = np.r_[np.random.randn(20, 2) - [2,2], np.random.randn(20,2) + [2,2]]
    y = [-1] * 20 + [1] * 20
    x = torch.FloatTensor(x)
    x = Variable(x, requires_grad = True)

    y = torch.FloatTensor(y)
    y = y.view(-1,1)
    y = Variable(y)
    Model = SVM()

    #optimizer = torch.optim.Adam(Model.parameters(), lr = 1e-2)
    # Way1 to add the L2-Norm penalty
    optimizer = torch.optim.Adam(Model.parameters(), lr = 1e-2, weight_decay = 0.01)
    for i in range(iteration):
        y_pred = Model(x)
        
        L2_loss = 0
        # Way2 to add the L2-Norm penalty
        '''
        for param in Model.parameters():
            L2_loss += torch.sqrt(torch.sum(torch.pow(param, 2)))
        '''
        # Training
        loss = torch.sum(torch.clamp(1-y*y_pred, min=0)) + 0.01 * L2_loss
        if i % 100 == 0:
            print('Epoch {}, Loss = {}'.format(i, loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Fetch the parameter in model to plot the figure
    parameter = []
    for para in Model.parameters():
        parameter.append(para.data.detach().numpy())
    
    x_d = x.detach().numpy()
    for i in range(20):
        plt.scatter(x_d[i][0], x_d[i][1], color='b')
    for i in range(20,40):
        plt.scatter(x_d[i][0], x_d[i][1], color='r')
    xx = np.linspace(-5,5)
    yy = []
    for xx_pro in xx:
        # Plot the SVM line
        yy.append((0.5 - xx_pro * parameter[0][0][0] - parameter[1][0]) / parameter[0][0][1])
    plt.plot(xx, yy, 'k-')
    plt.show()   

def Q15():
    print('The Question 15')
    # The linear regression model
    x = [3, 4, 5, 6, 7, 8, 9]
    x_d = np.array(x)
    x = np.array(x).reshape(-1,1)
    y = []
    xx_d = np.linspace(2,10)
    xx = np.linspace(2,10).reshape(-1,1)
    
    for i in x:
        y.append(i * 2 + random.normalvariate(0,1))
    reg = LinearRegression().fit(x, y)
    plt.scatter(x, y, color='b')
    yy = reg.predict(xx)
    plt.plot(xx, yy, 'k-')
    plt.show()
    
    # The Logistic regression model
    x_d = np.linspace(2,8) 
    y_d = [0 for i in range(25)] + [1 for i in range(25)]
    x = np.linspace(2,8).reshape(-1,1) 
    y = np.array(y_d).reshape(-1,1)
    xx = np.linspace(2,10)
    x_d = np.array(x)
    clf = LogisticRegression(random_state=0).fit(x,y)
    yy = clf.predict_proba(x)[:,1]
    plt.scatter(x, y, color='b')
    plt.plot(xx, yy, 'k-')
    plt.show()
    # The SVM model
    random.seed(0)
    x = np.r_[np.random.randn(20, 2) - [2,2], np.random.randn(20,2) + [2,2]] # Add Noise
    y = np.array([-1] * 20 + [1] * 20)

    clf = SVC(kernel = 'linear')
    clf.fit(x, y)
    coef = clf.coef_[0]
    a = - coef[0] / coef[1]
    xx = np.linspace(-5, 5)

    # plot the separation line
    yy = a * xx - (clf.intercept_[0] / coef[1])
    # plot the lower separation line
    b = clf.support_vectors_[0]
    y_d = a * xx + (b[1] - a * b[0])
    # plot the upper separation line
    b = clf.support_vectors_[-1]
    y_u = a * xx + (b[1] - a * b[0])
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, y_d, 'k--')
    plt.plot(xx, y_u, 'k--')
    # circle the points 
    plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Paired)
    plt.show()

def Q17_20():
    print('The Question 17-20')
    # Create the class to load the dataset
    class CIFAR10_Load(torch.utils.data.dataset.Dataset):
        def __init__(self, root, train = True):
            self.root = root
            self.train = train
            data = []
            labels = []
            if self.train:
                file_list = [self.root + '\\data_batch_' + str(i) for i in range(1,6)]
            else:
                file_list = [self.root + '\\test_batch']
            for file in file_list:
                with open(file, 'rb') as fo:
                    pack_dict = pickle.load(fo, encoding='bytes')
                    data += [pack_dict[b'data']]
                    labels += [pack_dict[b'labels']]
            # Transfer the list into numpy array
            data = np.concatenate(data)  
            labels = np.concatenate(labels)
            # Transfer the data into (batch_size, 32, 32, 3) so that we can visualize it
            self.data = np.transpose(np.reshape(data,[-1, 3, 32, 32]),[0, 2, 3, 1])        
            self.labels = labels
        
        def __len__(self):
            return  self.data.shape[0]
        def __getitem__(self,idx):
            return self.data[idx], self.labels[idx]
    # Load the data
    train = CIFAR10_Load(root = 'D:\Documents\课程介绍\神经网络与深度学习\Project1\cifar-10-python\cifar-10-batches-py', train= True)
    test = CIFAR10_Load(root = 'D:\Documents\课程介绍\神经网络与深度学习\Project1\cifar-10-python\cifar-10-batches-py', train= False)
    
    TrainLoader = torch.utils.data.DataLoader(train, batch_size=5000, num_workers=0,pin_memory=False)

    print('Start Processing')
    time1 = time.time()
    total = []
    for X_batch, y_batch in TrainLoader:
        # Question 16
        '''
        plt.imshow(X_batch[1])
        plt.show()
        '''
        
        # Question 20
        Mean = np.mean(np.reshape(X_batch.numpy(), [-1,3]), axis = 0).tolist()
        Std = np.std(np.reshape(X_batch.numpy(), [-1,3]), axis = 0).tolist()
        print('Mean of CIFAR-10’ training set within each RGB channel is: ', Mean)
        print('Std of CIFAR-10’ training set within each RGB channel is: ', Std)
    # Question 19
    time2 = time.time() - time1
    print(time2)

def Q18():
    img = Image.open('D:\Documents\课程介绍\神经网络与深度学习\Project1\picture.jpg')
    
    transform=transforms.Compose([
                              transforms.Scale([1024,1550]),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[2,3,2], std=[2, 3, 2])])
                             

    img2 = transform(img)
    img_2 = transforms.ToPILImage()(img2).convert('RGB')
    img_2.show()

def Q21():
    def process_bar(percent, start_str='', end_str='', total_length=0):
    # print the process bar
        bar = ''.join(["\033[31m%s\033[0m"%'-'] * int(percent * total_length)) + '>'
        bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
        print(bar, end='', flush=True)
    print('Question 21 is processing.')
    def transfer(r,g,b):
        blank = 50
        gap = (255 - blank) / 26
        # the piecewise linear transformation function
        j = 0.2126 * r + 0.7152 * g + 0.0722 * b
        character = ''
        if j < blank:
            character = ' ' 
        else:
            character = chr(65 + int((j-blank)/gap))
            if character == 'Z':
                character = ' '
        return character
    img = Image.open("D:\Documents\课程介绍\神经网络与深度学习\Project1\pic.jpg")
    pix = img.load()
    width = img.size[0]
    height = img.size[1]
    new_pic = Image.new("RGB", (width, height))
    new_arr = [['' for i in range(width)] for j in range(height)]
    
    for x in range(width):
        for y in range(height):
            r, g, b = pix[x, y][0:3]
            n = transfer(r,g,b)
            # transfer the pixel in different cernel into a new one
            new_arr[y][x] = n
            #f.write(n)
        #f.write('\n')
        time.sleep(0.1)
        end_str = '100%'
        process_bar(x/width, start_str='', end_str=end_str, total_length=15)
    f = open("D:\Documents\课程介绍\神经网络与深度学习\Project1\Test.txt", 'w')
    for i in range(height):
        for j in range(width):
            f.write(new_arr[i][j])
        f.write('\n')
    f.close()

def Q22():
    # item 1
    origin = np.random.normal(0,1,(10,2))
    r = []
    Angle = []
    for i in range(10):
        coor = [0, 0]    
        coor[0] = origin[i,0]
        coor[1] = origin[i,1]
        r = np.sqrt(coor[0]*coor[0] + coor[1]*coor[1])
        Angle = np.arctan(coor[1]/coor[0]) * 180 / math.pi
        # Dealing with different quadrants
        if coor[0] > 0 and coor[1] > 0:
            print(origin[i,:],'-->', (r, Angle))
        elif coor[0] < 0 and coor[1] > 0:
            print(origin[i,:],'-->', (r, Angle+180))
        elif coor[0] < 0 and coor[1] < 0:
            print(origin[i,:],'-->', (r, Angle+180))
        else:
            print(origin[i,:],'-->', (r, 360 + Angle))

    # item 2
    class Symmetric(np.ndarray):
        def __setitem__(self, index, value):
            i = index[0]
            j = index[1]
            # Rewrite the setitem function
            super(Symmetric, self).__setitem__((i,j), value)
            super(Symmetric, self).__setitem__((j,i), value)
    def generate(shape):
        # generate a symmetric
        result = np.zeros(shape).view(Symmetric)
        for i in range(shape[0]):
            for j in range(i,shape[0]):
                result[i,j] = random.randint(1,10)
        return result
    
    a = generate((5,5))
    print(a)
    # Test whether we can change the value symmetrically
    a[1,0] = 10
    print(a)
    # item 3 
    def distance_point(a,b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    def distance(p0,p1,p2):
        # Heron's formula
        l1 = distance_point(p0,p1)
        l2 = distance_point(p1,p2)
        l3 = distance_point(p2,p0)
        p = (l1 + l2 + l3) / 2
        s = np.sqrt(p * (p - l1) * (p - l2) * (p - l3))
        return 2 * s / l1
    
    P0 = np.random.uniform(-5, 5, (5,2))
    P1 = np.random.uniform(-5, 5,(5,2))
    p = np.random.uniform(-5, 5, (5,2))
    for i in range(5):
        dist = distance(P0[i,],P1[i,],p[i,])
        print(p[i,],'--> (',P0[i,],P1[i,],') = ',dist)
    

def Q23():
    print('The Question 23') 
    A = [[110, 120,130], [210, 220, 230], [310, 320, 330]]
    coord = eval(input('Please input a coordinate: '))
    d1 = coord[0] - math.floor(coord[0])
    d2 = coord[1] - math.floor(coord[1]) 
    f1 = math.floor(coord[0])
    f2 = math.floor(coord[1])
    result = A[f1-1][f2-1] * (1-d1) * (1-d2) + \
             A[f1][f2-1] * d1 * (1-d2) + \
             A[f1-1][f2] * (1-d1) * d2 + \
             A[f1][f2] * d1 * d2
    print('BilinearInterpolation(A,',coord,') == ', result)

def Q24():
    print('The Question 24') 
    def descartes(v,n):
        # Calculate the Cartesian product between 2 lists
        return [v[i]+[n[j]]  for i in range(len(v)) for j in range(len(n)) ]
    # Input the data
    l = input('Please input the vectors: ')
    l = l.lstrip('[').rstrip(']').split('], [')
    origin = []
    for s in l:
        s = s.split(',')
        ints = []
        for ch in s:
            ints.append(int(ch))
        origin.append(ints)
    result = [[i] for i in origin[0]]
    for i in range(1,len(origin)):
        result = descartes(result, origin[i])
    print(result)

def Q25():
    print('The Question 25')
    def subpart(Z,shape,fill,position):
        b_coord = (position[0] - math.ceil(shape[0]/2), position[1] - math.ceil(shape[1]/2))
        # Navigate to start position
        matrix = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                write_i = i + b_coord[0]
                write_j = j + b_coord[1]
                # Padding
                if (write_i<0) or (write_j<0) or (write_i>=len(Z)) or (write_j>=len(Z[i])):
                    row.append(0)
                else:
                    row.append(Z[write_i][write_j])
            matrix.append(row)
        return matrix
        
    print('The Question 25')
    Z = np.random.randint(0,10,(5,5))
    print(Z)
    shape = (4,4)
    fill = 0
    position = (1,1)
    result = subpart(Z,shape,fill,position)
    for i in range(len(result)):
        print(result[i])
    
def Q26():
    print('The Question 26')
    def add(A,B):
        row_len = len(A)
        column_len = len(B[0])
        cross_len = len(B)
        result = [[0 for j in range(column_len)] for i in range(row_len)]          
        for i in range(row_len):
            for j in range(column_len):
                result[i][j] = A[i][j] + B[i][j]
        return result
    
    def subtract(A,B):
        row_len = len(A)
        column_len = len(B[0])
        cross_len = len(B)
        result = [[0 for j in range(column_len)] for i in range(row_len)]          
        for i in range(row_len):
            for j in range(column_len):
                result[i][j] = A[i][j] - B[i][j]
        return result

    def scalar_multiply(A,B):
        row_len = len(A)
        column_len = len(A[0])
        result = [[0 for j in range(column_len)] for i in range(row_len)]          
        for i in range(row_len):
            for j in range(column_len):
                result[i][j] = A[i][j] * B
        return result

    def multiply(A,B):
        row_len = len(A)
        column_len = len(B[0])
        cross_len = len(B)
        result = [[0 for j in range(column_len)] for i in range(row_len)] 
        for i in range(row_len):
            for j in range(column_len):
                for k in range(cross_len):
                    temp = A[i][k] * B[k][j]
                    result[i][j] += temp	
        return result
    
    def identity(n):
        result = [[0 for j in range(n)] for i in range(n)] 
        for i in range(n):
            result[i][i] = 1
        return result
    
    def transpose(A):
        row_len = len(A)
        column_len = len(A[0])
        result = [[[0] for j in range(row_len)] for i in range(column_len)]
        for i in range(column_len):
            for j in range(row_len):
                result[i][j] = A[j][i]
        return result

    def inverse(A):
        def det(mat):
            # Calculate the determinant of the metrix
            if len(mat) <= 0:
                return None
            if len(mat) == 1:
                return mat[0][0]
            else:
                total = 0
                for i in range(len(mat)):
                    n = [[row[r] for r in range(len(mat)) if r != i] for row in mat[1:]]
                    if i % 2 == 0:
                        total += mat[0][i] * det(n)
                    else:
                        total -= mat[0][i] * det(n)
                return total      
        row_len = len(A)
        column_len = len(A[0])
        det_all = det(A)
        result = [[[0] for j in range(row_len)] for i in range(column_len)]
        for i in range(row_len):
            for j in range(column_len):
                # Generate the Adjoint matrix
                cofactor = [[A[b][a] for a in range(column_len) if a != j] for b in range(row_len) if b != i]
                result[j][i] = (-1) ** (i+j) * det(cofactor) / det_all
        return result

    matrix_a = [[12, 10], [3, 9]] 
    matrix_b = [[3, 4], [7, 4]] 
    matrix_c = [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]] 
    matrix_d = [[3, 0, 2], [2, 0, -2], [0, 1, 1]] 
    print('add(matrix_a, matrix_b) == ',add(matrix_a, matrix_b)) 
    print('subtract(matrix_a, matrix_b) == ',subtract(matrix_a, matrix_b)) 
    print('scalar_multiply(matrix_b, 3) == ',scalar_multiply(matrix_b, 3))
    print('multiply(matrix_a, matrix_b) == ',multiply(matrix_a, matrix_b))
    print('identity(3) == ', identity(3)) 
    print('transpose(matrix_c) == ', transpose(matrix_c)) 
    print('inverse(matrix_d) == ', inverse(matrix_d))

def Q27(a,b):
    print('The Question 27')
    p = max(abs(a),abs(b))
    q = min(abs(a),abs(b))
    if q == 0:
        result = p
        print('GCD(',a,',',b,') =',p)
        return 
    temp = p % q
    while (temp != 0):
        p = q
        q = temp
        temp = p % q
    print('GCD(',a,',',b,') =',q)
    return

def Q28(N):
    print('The Question 28')
    i = 0
    while i < int(math.sqrt(2*N)):
        i += 1
        if (2 * N / i + 1 - i) % 2 == 0:
            start = int((2 * N / i + 1 - i) / 2)
            print(list(range(start,start+i)))
    return 

def Q29():
    print('The Question 29')
    char_list = input('Please input: ')
    char_list = char_list.split(',')
    legal = []
    for s in char_list:
        if 6 <= len(s) <= 12:
            char = re.findall(r'[a-z]', s)
            upperchar= re.findall(r'[A-Z]', s)
            num = re.findall(r'[0-9]', s)
            spe = re.findall(r'[$#@]', s)
            if char and upperchar and num and spe:
                legal.append(s)
    string = ','.join(legal)
    print(string)

if __name__ == "__main__":
    # You can comment out the program what you don't need to test
    #Q1()
    #Q2()
    #Q3()
    #Q4()
    #Q5()
    #Q6()
    #Q7()
    #Q8()
    #Q9()
    #Q10()
    #Q11()
    #Q12()
    #Q13_14()
    #Q15()
    #Q17_20()
    #Q18()
    #Q21()
    #Q22()
    #Q23()
    #Q24()
    #Q25()
    Q26()
    '''
    Q27(3,5)
    Q27(6,3)
    Q27(-2,6)
    Q27(0,3)
    '''
    #Q28(1000)
    #Q29()