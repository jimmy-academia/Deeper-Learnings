<pre>
List
List[a:b:c] 有a ~ 沒有b, a, a+c, a+2c...

L = list(range(1,21))   //list 轉換成序列  
L = L[ i*10 for i in L ]

list(map(f, iterable)) //for python 3
is basically equivalent to:  [f(x) for x in iterable]

list:
a = [1,2,3,4]   a.append(b) .push .pop ....

del a[a:b:c] 去除這些元素

兩個list 可以用zip:
a = [1,2,3]
b = (4,5,6)
for i,j in zip(a,b): print i+j

for index, item in enumerate(a)

dictionary
d = {‘a’: 1, ‘b’:2, ‘c’:3, ‘d’:4}
d = dict(a = 1, b = 2, c = 3)
    d[‘d’] = 4
    或是用append
    .append(dict(firstName=’Al’, lastName= ‘fellow’))
d = {x : x**2 for x in range(1,5)}
d = dict((key, value) for key, value in d.items() if value <= 1)
sum(d.values())

>> to json: 
import json
with open("company1.json", "w") as file:
    json.dump(d, file, indent=4, sort_keys=True) // d is the dictionary
<<from json:
with open("company1.json","r") as file:
    d = json.loads(file.read()) // d is dictionary



print
from pprint import pprint //for dictionary
print (“i am %s written in %s” %(first, second))
use %d for integer

    new way: ‘{},{}’.format(a,b)    https://pyformat.info

function

global c //define in function can be used outside

string
a = ‘asefm bame smf’
a.split(‘ ‘) //用 ‘ ’ 來切割變成 list of string
string.replace(“,“ , ” “ )   //replace comma with space
“\n” is a new line
if letter in ‘python’: 可對照string 裡的單字

characters = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*("
chosen = random.sample(characters, 6)  // chosen = [“xx”, “xx”, .....]
password = "a".join(chosen)  // join 完得 ”xxaxxaxxaxxa.....”


open; os
 with open(filepath, 'r') as file: //’r’ 代表read, 其他: w,x,...
        strng = file.read()     // a string

with open("letters.txt", "w") as file:
    for letter in string.ascii_lowercase:
        file.write(letter + "\n")

file.seek(0) //回到文件頂點

import os
if not os.path.exists("letters"):
    os.makedirs("letters")

可用 glob 讀一系列檔案
import glob
file_list = glob.glob("letters/*.txt")
for filename in file_list:
    with open(filename, "r") as file:
        letters.append(file.read().strip("\n"))
或是簡單用for loop(比較不好，因為針對黨名是英文字母設計)
import string
a = []
for i in string.ascii_lowercase:
  with open("./letters/" + i +".txt", 'r') as file:
    a.append(file.read())
print(a)

a = input(“some messages”) //a = 使用者輸入的東西(string)



math
import math
math.sqrt(9)
dir(math) //查math 裡有甚麼 --> 有 exp, cos, cosh, pi, log, e, ...
help(math.pow) //查 pow 怎麼用


time
import time
time.sleep(2) //停2秒

for/ while loop --> break
if --> pass
continue: ignore and back to loop
...
try:
        return d[word]
except KeyError:
        return "That word does not exist."

requests
import requests as re
r = re.get(“url”)
r.text.count(‘a’)

webbrowser.open_new(url) 
url = “https://www.youtube.com/results?search_query=%s” % str(something)


進階!!


Generator

def fib():
    for i in range(2):
        yield 1
    a = 1
    b = 1    
    for i in range(3,20):
        a = a + b
        a,b = b,a
        yield  b


function
def foo(a, b, c, *rest, **option):
    print("And all the rest... %s" % list(therest))
    if options.get("action") == "sum":
        //do something

numbers = [1,2,4,-1,-4,-3]
newlist = [int(a) for a in numbers if a > 0]

try except
def get_last_name():
    try: 
        return actor["name"].split()[1]
    except IndexError:
        return "NAN"
typical errors

set
a = set(['a','b','c']) = set(somelist)
b = set(['c', 'd'])

a.intersection(b) == b.intersection(a) == {'c'}
a.symmetric_difference(b) == b...a. == {'a','b','d'}
a.difference(b) = a-b
b....a = b-a
a.union(b)


decoration
@xdfawej// 對下一個函數做事??
def xxx()

</pre>