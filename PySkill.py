

a = [1,2,3,4,5,6,7,8]
for i, item in enumerate(a):
    print(i, item)

for i, item in enumerate(a, 1):
    print(i, item)

b = list(enumerate('abc'))
print(b)

c = list(enumerate('abc', 1))
print(c)

#------------------------------------------
# python has no xrange, only range
from pprint import pprint
dict1 = {i : i * i for i in range(10)}
print(dict1)
pprint(dict1)

set1 = {i * 2 for i in range(10)}
print(set1)

def xrange(x):
    n=0
    while n<x:
        yield n
        n+=1

# 强制浮点除法
result = 1 / 2
print(result)

# 简单服务器  你是否想要快速方便的共享某个目录下的文件呢？你可以这么做:python3 -m http.server 这样会为启动一个服务器。

# 对Python表达式求值
# 我们都知道eval函数，但是我们知道literal_eval函数么？也许很多人都不知道吧。可以用这种操作：
import ast
expr = "[1, 2, 3]"
my_list = ast.literal_eval(expr)
print(my_list)

# 来代替以下这种操作：
my_list = eval(expr)
print(my_list)

# 脚本分析
# 你可以很容易的通过运行以下代码进行脚本分析：
# python -m cProfile my_script.py

'''
对象自检
在Python 中你可以通过dir() 函数来检查对象。正如下面这个例子：
>>> foo = [1, 2, 3, 4]
>>> dir(foo) 
['__add__', '__class__', '__contains__', 
'__delattr__', '__delitem__', '__delslice__', ... , 
'extend', 'index', 'insert', 'pop', 'remove', 
'reverse', 'sort']


调试脚本
你可以很方便的通过pdb模块在你的脚本中设置断点。正如下面这个例子：
import pdb
pdb.set_trace()
你可以在脚本的任何地方加入pdb.set_trace()，该函数会在那个位置设置一个断点。

if 结构简化
如果你需要检查几个数值你可以用以下方法：
if n in [1,4,5,6]:
来替代下面这个方式：
if n==1 or n==4 or n==5 or n==6:


字符串/数列 逆序
你可以用以下方法快速逆序排列数列：
>>> a = [1,2,3,4]
>>> a[::-1]
[4, 3, 2, 1]
 
# This creates a new reversed list. 
# If you want to reverse a list in place you can do:
a.reverse()

这总方式也同样适用于字符串的逆序：
>>> foo = "yasoob"
>>> foo[::-1]
'boosay'


优美地打印
你可以通过以下方式对字典和数列进行优美地打印：
from pprint import pprint 
pprint(my_dict)
这种方式对于字典打印更加高效。此外，如果你想要漂亮的将文件中的json文档打印出来，你可以用以下这种方式：
cat file.json | python -m json.tool

三元运算
三元运算是if-else 语句的快捷操作，也被称为条件运算。这里有几个例子可以供你参考，它们可以让你的代码更加紧凑，更加美观。
[on_true] if [expression] else [on_false]
x, y = 50, 25
small = x if x < y else y

'''

x, y = 50, 25
small = x if x < y else y
print(small)

