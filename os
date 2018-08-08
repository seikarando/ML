在windows下使用python自带的gui shell来测试脚本，有时候我们需要进行如：切换/改变当前工作路径、显示当前目录、删除文件等。
所以，这些切换目录等操作都需要调用python的os 模块里的相关函数如下：
os.sep可以取代操作系统特定的路径分隔符。windows下为 “”
os.name字符串指示你正在使用的平台。比如对于Windows，它是'nt'，而对于Linux/Unix用户，它是'posix'。
os.getcwd()函数得到当前工作目录，即当前Python脚本工作的目录路径。
os.getenv()获取一个环境变量，如果没有返回none
os.putenv(key, value)设置一个环境变量值
os.listdir(path)返回指定目录下的所有文件和目录名。
os.remove(path)函数用来删除一个文件。
os.system(command)函数用来运行shell命令。
os.linesep字符串给出当前平台使用的行终止符。例如，Windows使用'rn'，Linux使用'n'而Mac使用'r'。
os.path.split(p)函数返回一个路径的目录名和文件名。
os.path.isfile()和os.path.isdir()函数分别检验给出的路径是一个文件还是目录。
os.path.exists()函数用来检验给出的路径是否真地存在
os.curdir:返回当前目录（'.')
os.chdir(dirname):改变工作目录到dirname
os.path.getsize(name):获得文件大小，如果name是目录返回0L
os.path.abspath(name):获得绝对路径
os.path.normpath(path):规范path字符串形式
os.path.splitext():分离文件名与扩展名
os.path.join(path,name):连接目录与文件名或目录
os.path.basename(path):返回文件名
os.path.dirname(path):返回文件路径

如在python 的gui shell下，要切换目录，命令如下：
>>> import os //导入os模块
>>> os.getcwd() //用os.getcwd()函数，来查看当前目录
结果显示：'C:\\Python27\\'
>>>os.chdir('C:\\Python27\\Tools\\setuptools-1.3\\') //调用 os.chdir()函数切换到C:\Python27\Tools\setuptools-1.3\目录
注意：
不是chrdir，网上的有一个人笔误了，结果其他人也不验证，照着复制转载，将原作者的笔误给“继承了”。
因为是windows下使用，所以在每个斜杠“\\”要替换成“\\”双斜杠，因为要转义字符

切换后，再次查看我们的当前目录：
>>> os.getcwd() //用os.getcwd()函数，来查看当前目录
结果显示：'C:\\Python27\\Tools\\setuptools-1.3' //说明已经成功切换了