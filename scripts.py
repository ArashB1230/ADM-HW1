"""A lot of the solutions for problem 1 are not my own work.
I am writing to gain a better understanding. However,
problems 2 and 3 are my own work.2 and 3 are my own work.
"""
#Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

#Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 1:
    print("Weird")
elif n % 2 == 0:
    if 1<n<6:
        print("Not Weird")
    if 5<n<21:
        print("Weird")
    if 20<n:
        print("Not Weird")
    
#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(0,n):
        print(i**2)

#Write a function
def is_leap(year):
    leap = False
    if year % 400 ==0:
        leap=True
    elif year%100 !=0 and year%4==0:
        leap=True
    else:
        leap=False
    # Write your logic here
    
    return leap


#Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i,end="")

#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
"""
using for loop
fav_list=[]
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if i+j+k!=n:
                fav_list.append([i,j,k])
print(fav_list)"""

fav_list=[
[i,j,k] for i in range(x+1)
for j in range(y+1)
for k in range(z+1) 
if i+j+k!=n
]
print(fav_list)

#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    r=set()
    for i in arr:
        r.add(i)
    r.remove(max(r))
    print(max(r))

#Nested Lists
S=[]
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        S.append([name,score])

student_score=sorted(list(set([s[1]for s in S])))
second_lowest_grade=student_score[1]
all_second_lowest=[s for s in S if s[1]==second_lowest_grade]
all_second_lowest.sort()
for i in all_second_lowest:
    print(i[0])

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    #s_m: student marks
    s_m = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        s_m[name] = scores
    query_name = input()
    av_m=s_m[query_name]
    # length of marks=3 and ".nf":= n places after decimal 
    print(format(sum(av_m)/3, '.2f'))

#Lists
if __name__ == '__main__':
    N = int(input())
list=[]
for i in range (N):
    commands=input().split()
    if commands[0]=="insert":
        list.insert(int(commands[1]),int(commands[2]))
    elif commands[0]=="print":
        print(list)
    elif commands[0]=="remove":
        list.remove(int(commands[1]))
    elif commands[0]=="append":
        list.append(int(commands[1]))
    elif commands[0]=="sort":
        list.sort()
    elif commands[0]=="pop":
        list.pop()
    else:
        list.reverse()

#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))

#sWAP cASE
def swap_case(s):
    '''
    S = ""
    for char in s:
        if char.isupper():
            S += char.lower()
        elif char.islower():
            S += char.upper()
        else:
            S += char
    return s '''
    return s.swapcase()
    
#String Split and Join


def split_and_join(line):
    # write your code here
    s=line.split()
    t="-".join(s)
    return(t)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print("Hello " + first +" "+ last + "! You just delved into python." )
    # Write your code here

#Mutations
def mutate_string(string, position, character):
    return string[:position] + character+string[position+1:]

#Find a string
def count_substring(string, sub_string):
    counter=0
    for i in range (len(string)):
        s=string[i:i+len(sub_string)]
        if s==sub_string:
            counter=counter + 1
    return counter


#String Validators
if __name__ == '__main__':
    s = input()
    a,b,c,d,e = False,False,False,False,False
    
    for i in s :
        if i.isalnum():
            a = True
        if i.isalpha():
            b = True
        if i.isdigit():
            c = True
        if i.islower():
            d = True
        if i.isupper():
            e = True        
    print(a);print(b);print(c);print(d);print(e)   

#Text Wrap


def wrap(string, max_width):
    return textwrap.fill(string, max_width)

#Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = input()
list1 = n.split()
h, l = int(list1[0]), int(list1[1])
i = 0
while i < h // 2:
    mask = '.|.' * (i * 2 + 1)
    print(mask.center(l, '-'))
    i += 1
print('WELCOME'.center(l, '-'))
i = h // 2 - 1
while i >= 0:
    mask = '.|.' * (i * 2 + 1)
    print(mask.center(l, '-'))
    i -= 1

#Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    l = []
    for i in range(n):
        subs = "-".join(alphabet[i:n])
        p = subs[::-1] + subs[1:]
        l.append(p)
    max_width = len(l[0])
    for i in range(n - 1, 0, -1):
        print(l[i].center(max_width, '-'))
    for i in range(n):
        print(l[i].center(max_width, '-'))

#Introduction to Sets
def average(array):
    # your code goes here
    return (sum(set(array)))/(len(set(array)))
    
#Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
M=int(input())
m=set(map(int, input().split()))

N=int(input())
n=set(map(int, input().split()))
result=m.symmetric_difference(n)

for r in sorted(result):
    print(r)

#Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
countryname=set()
for i in range (n):
    countryname.add(input())
print(len(countryname))

#Set .discard(), .remove() & .pop()
n=int(input())
s=set(map(int,input().split()))
commands=int(input())
for i in range (commands):
    cmd=input().split()
    if cmd[0]== "remove":
        s.remove(int(cmd[1]))
    if cmd[0]== "discard":
        s.discard(int(cmd[1]))
    if cmd[0]== "pop":
        s.pop()
summ=0
for i in s:
    summ= summ+ i
print(summ)
    
#Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
A=set(map(int,input().split()))
m=int(input())
B=set(map(int,input().split()))
print(len(A.union(B)))

#Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
A=set(map(int,input().split()))
m=int(input())
B=set(map(int,input().split()))
print(len(A.intersection(B)))

#Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
A=set(map(int,input().split()))
m=int(input())
B=set(map(int,input().split()))
print(len(A.difference(B)))

#Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
A=set(map(int,input().split()))
m=int(input())
B=set(map(int,input().split()))
print(len(A.symmetric_difference(B)))

#Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s = set(map(int, input().split()))
cmds = int(input())

for _ in range(cmds):
    c, *args = input().split()
    t = set(map(int, input().split()))
    if c == 'update':
        s.update(t)
    elif c == 'intersection_update':
        s.intersection_update(t)
    elif c == 'difference_update':
        s.difference_update(t)
    elif c == 'symmetric_difference_update':
        s.symmetric_difference_update(t)
print(sum(s))

#The Captain's Room

# Enter your code here. Read input from STDIN. Print output to STDOUT
k = int(input())

t = input().split()

# convert it to int list
for i in range(len(t)):
    t[i] = int(t[i])
s = (sum(set(t)) * k - sum(t))    
print(s // (k - 1))

#Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
for i in range(n):
    a = int(input())
    set_A = set(map(int, input().split()))
    b = int(input())
    set_B = set(map(int, input().split()))
    if len(set_A - set_B) == 0:
        print("True")
    else:
        print("False")

#Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
a = set(input().split())
c, d = 0, 0

n = int(input())
for i in range(n):
    data = set(input().split())
    if a.issuperset(data):
        c += 1
    else:
        d += 1
if d != 0:
    print('False')
else:
    print('True')

#collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
n=int(input())
s=list(map(int, input().split(' ')))
Dict=Counter(s)
x=int(input())
p=0
for i in range (x):
    size,price=map(int,input().split(' '))
    if Dict[size]:
        Dict[size]-=1
        p=p+price
print(p)
    
#DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
d=defaultdict(list)
list1=[]
m,n=map(int, input().split())
for i in range (m):
    a=input()
    d[a].append(i+1)
for i in range (n):
    b=input()
    list1=list1+[b]
for i in list1:
    if i in d:
        print(' '.join(map(str,d[i])))
    else:
        print("-1")

#Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
n=int(input())
data= namedtuple("data",input())
s=sum([int(data(*input().split()).MARKS)for i in range(n)])/n
print(s)

#Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict

od = OrderedDict()
for i in range(int(input())):
    j,k = input().rsplit(' ', 1)
    if j in od:
        od[j] = int(od[j])+ int(k)
    else:
        od[j]=k
for j,k in od.items():
    print(j,k)
    
#Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter, OrderedDict
class OrderedCounter (Counter, OrderedDict):
    pass
d  = OrderedCounter(input() for _ in range(int(input())))
print(len(d))
print(*d.values())

#Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque

d = deque()
n=int(input())
for i in range(n):
    cmd=input().split()
    if cmd[0] == "append":
        d.append(cmd[1])
    elif cmd[0] == 'pop':
        d.pop()
    elif cmd[0] == "appendleft":
        d.appendleft(cmd[1])
    elif cmd[0] == "popleft":
        d.popleft()
  
print(*d)

#Company Logo
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter
class OrederedCounter(Counter):
    pass

if __name__ == '__main__':
    [print(*c) for c in OrederedCounter(sorted(input())).most_common(3)]
   # s = input()

#Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
for t in range (int(input())):
    input()
    lst=[int(i) for i in input().split()]
    min_list=lst.index(min(lst))
    left=lst[:min_list]
    right=lst[min_list+1:]
    if left==sorted(left , reverse=True)and right==sorted(right):
        print("Yes")
    else:
        print("No")

#Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
weekdays=["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
m,d,y=map(int, input().split())
print(weekdays[calendar.weekday(y,m,d)])

#Time Delta
#!/bin/python3

import math
import os
import random
import re
import sys
import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    t1 = datetime.datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
    t2 = datetime.datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
    delta = t1 - t2
    return delta
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)
        delta = delta.total_seconds()
        if delta<0:
            delta = delta * (-1)    
        delta = int(delta)
        fptr.write(str(delta) + '\n')

    fptr.close()

#Exceptions
    # Enter your code here. Read input from STDIN. Print output to STDOUT
for i in range(int(input())):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

#Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N,X=input().split()
io=list()
for _ in range(int(X)):
    ip=map(float,input().split())
    io.append(ip)
for i in zip(*io):
    print(sum(i)/len(i))

#Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys
from operator import itemgetter


if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input().strip())
    arr1=sorted(arr,key=itemgetter(k))
    for i in arr1:
        print(*i)

#ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
lower=""
upper=""
odd=""
even=""
s=sorted(input())
for i in (s):
    if i.islower():
        lower+=i
    elif i.isupper():
        upper+=i
    elif int(i)%2 !=0:
        odd+=i
    elif int(i)%2==0:
        even+=i
print(lower + upper + odd + even)  

#Map and Lambda Function
cube = lambda x: x ** 3

def fibonacci(n):
    # return a list of fibonacci numbers
    a=0
    b=1
    l=[]
    if n==0:
        pass
    elif n==1:
        l.append(a)
    else:
        l.append(a)
        l.append(b)
        for i in range (2,n):
            c=a+b
            a=b
            b=c
            l.append(c)
    return l

#XML 1 - Find the Score


def get_attr_number(node):
    # your code goes here
    v=0
    for i in node.iter():
        v=v+len(i.attrib)
    return v

#XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    if(level==maxdepth):
        maxdepth+=1
    for i in elem:
        depth(i,level+1)

#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

#Decorators 2 - Name Directory


def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
        # complete the function
    return inner

#Arrays

def arrays(arr):
    # complete this function
    # use numpy.array
     r = numpy.array(arr[::-1], float)
     return r

#Shape and Reshape
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
l=list(map(int,input().split()))
arr=np.array(l)
arr.shape=(3,3)
print(arr)

#Transpose and Flatten
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n,m=map(int, input().split())
list1=[]
for i in range(n):
    a=list(map(int,input().split()))
    list1.append(a)
arr=np.array(list1)
print(np.transpose(arr))
print(arr.flatten())

#Concatenate
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy

n, m, t = map(int, input().split())
arr_1 = [list(map(int, input().split())) for _ in range(n)]
arr_2 = [list(map(int, input().split())) for _ in range(m)]

print(numpy.concatenate((arr_1, arr_2)))

#Zeros and Ones
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
nums=tuple(map(int,input().split()))
print(np.zeros(nums,dtype=np.int))
print(np.ones(nums,dtype=np.int))

#Eye and Identity
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n,m=map(int,input().split())
np.set_printoptions(sign=" ")
print(np.eye(n,m))

#Array Mathematics
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
N,H=map(int,input().split())
p=np.array([list(map(int,input().split())) for _ in range(N)], int)
q=np.array([list(map(int,input().split())) for _ in range(N)], int)
print(np.add(p,q),np.subtract(p,q),np.multiply(p,q), np.floor_divide(p,q), np.mod(p,q), np.power(p,q) , sep="\n")

#Floor, Ceil and Rint
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
np.set_printoptions(legacy='1.13')
a=np.array(input().split(), float)
print(np.floor(a),np.ceil(a),np.rint(a), sep="\n")

#Sum and Prod
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n,m=map(int,input().split())
array=np.array([input().split() for i in range(n)], int)
print(np.prod(np.sum(array,axis=0)))

#Min and Max
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n,m=map(int,input().split())
list1=[]
for i in range(n):
    l=list(map(int,input().split()))
    list1.append(l)
arr=np.array(list1)
print(np.max(np.min(arr,axis=1)))

#Mean, Var, and Std
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n,m=map(int,input().split())
list1=[]
for i in range(n):
    l=list(map(int,input().split()))
    list1.append(l)
arr=np.array(list1)
np.set_printoptions(legacy="1")
print(np.mean(arr,axis=1))
print(np.var(arr,axis=0))
print(round(np.std(arr), 11))

#Dot and Cross
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n=int(input())
a=np.array([input().split()for i in range(n)],dtype=int)
b=np.array([input().split()for i in range(n)],dtype=int)
print(a.dot(b))

#Inner and Outer
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
a=np.array(input().split(),int)
b=np.array(input().split(),int)
print(np.inner(a,b))
print(np.outer(a,b))

#Polynomials
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
arr=np.array(input().split(),dtype=float)
print(np.polyval(arr,int(input())))

#Linear Algebra
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
n=int(input())
list1=[]
for i in range(n):
    l1=list(map(float,input().split()))
    list1.append(l1)
arr1=np.array(list1)
p=np.linalg.det(arr1)
print(round(p,2))
#####################################################################################
#Birthday Cake Candles
#Wronganswer but work in first case 
#!/bin/python3
'''
import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    counter1=0
    mymax=0
    for i in candles:
        for j in candles:
            if i >=j and candles.index(i) != candles.index(j):
                counter1+=1 
            if counter1==len(candles)-1:
                mymax=i
                break
    counter2=0
    for j in candles:
        if j ==mymax:
            counter2+=1
    return counter2
                
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
'''
#########################################################################################
#Number Line Jumps
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#


def kangaroo(x1, v1, x2, v2):
    if v1 - v2 ==0 and x1 - x2!=0:
        return "NO"
    else:
        jump = (x2 - x1)/(v1 - v2)
        if jump==int(jump) and jump>=0 and (v1 - v2)!=0:
            return "YES"
        else:
            return "NO"
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
#########################################################################################
#Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    mylist=[2]
    likes=2
    for i in range(n-1):
        likes=int((likes*3)/2)
        mylist.append(likes)
    return sum(mylist)
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
