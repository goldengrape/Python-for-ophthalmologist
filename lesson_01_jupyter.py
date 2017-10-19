
# coding: utf-8

# # CoCalc编程环境
# 
# ## 注册cocalc
# 在浏览器
# 地址栏内输入 cocalc.com 
# 现代的浏览器基本都可以使用. 已知Chrome, Safari, 甚至Android手机上的Chrome和iPad上的Safari都能够很好地支持, IE没有试过. 
# 不需要在本地安装任何东西. 
# 
# ![](./img/cocalc00.png)
# 点击 Run CoCalc
# 
# ![](./img/cocalc01.png)
# 随便填点什么注册一个免费账户就可以开始了. 免费账户的主要限制在于不能从远端服务器再去访问网络, 如果你需要编写一个网络服务的程序, 那么就不适用了. 对于本教程所需要的内容, 免费账户已经足够使用了. 
# 
# 
# 

# ## 建立cocalc project
# 
# 注册成功以后, 再进入Cocalc登录, 就可以到达项目管理的界面了, 一个Cocalc账户可以建立很多个项目(project), 你可以把一个project当作一台电脑, 如果觉得自己把一个project的设置搞乱了, 新建另外一个即可. 
# 
# 新建project就是起个名字, 然后点击 create new project
# ![](./img/cocalc02.png)

# ## 新建jupyter notebook
# 
# 进入project以后, 会默认处于文件管理器的状态, 此时可以打开以前建立的文件, 或者新建文件. 
# 现在, 我们新建一个jupyter文件, 请点击Create按钮
# ![](./img/cocalc03.png)
# 
# 在cocalc里面支持很多中文件的编辑, 其中我们主要使用的是jupyter notebook, 这是一种python文件的变体, 在科学计算领域很流行. 本课程就是以jupyter为基础进行讲解的. 
# 
# ![](./img/cocalc04.png)
# 
# 点击了create按钮以后, 会发现展开了很多按钮, 先随便起个文件名, 比如lesson1, 不需要写上后缀, 然后点击jupyter notebook这个按钮, 就可以新建一个jupyter notebook了. 
# 
# 如果已经是自己编辑好的文件, 或者是一些需要处理的数据文件, 也可以使用下面的upload部分来上传文件. 
# 
# 新建文件中, 除了jupyter notebook, 还有: 
# * sage worksheet: 这是Sage版本的"notebook", sage是一个与python/ matlab / mathematica很类似的数学计算语言, 如果你需要随手解个简单的方程, 求导, 求积分之类的符号运算, sage都是不错的选择. 
# * markdown: 相当于一种使用纯文本编写的word文件, 形式是纯文本的, 但使用简单的符号来标记格式, 比word高级多了. 不过Jupyter notebook里面也支持markdown, 所以一般我就直接用jupyter了. 
# * LaTeX: 一流大学理工科学生写论文的时候应该是要用LaTeX的, 编辑数学公式很简单, markdown/ jupyter里面也可以借用LaTeX的部分命令来撰写数学公式. 
# * Terminal: cocalc里很有意思的是把命令行终端程序也当作一个文件来处理, 如果要进入Linux命令行工具, 那么就要新建一个terminal文件, 然后打开这个terminal文件就可以使用Linux命令了. cocalc是基于Linux的, 一些常用的Linux命令是很容易在网上找到的, 比如列出文件ls, 跳转目录cd, 删除文件rm, 如果要在命令行里运行python, 就直接输入python, 或者python3即可. 
# 

# ## Jupyter界面
# 
# ![](./img/cocalc05.png)
# 
# 1. 菜单: 跟常规的菜单一样, 保存, 新建之类的命令都在里面, 里面的Help也很有帮助
# 2. 运行按钮: Jupyter里面是一个一个cell, 要运行一个cell的内容, 可以用这个按钮
# 3. cell的类型: 可以是Code, 也可以是Markdown. 可以是需要运行的代码, 也可以是用来说明的文字. 
# 4. In[ ]: 后面的空格就是可以输入的代码或者文字的地方. 
# 5. 如果对cocalc有什么问题, 可以用help按钮, 给后台发email询问, 他们的服务真的很棒, 经常是秒回. 
# 

# ## markdown与数学公式
# 
# 我们先来写一段Markdown文本: 
# 先把In[ ] 的形式改成Markdown: 
# ![](./img/cocalc06.png)
# 
# 这时输入框前面的In[ ]标记没了, 输入框也变成Type markdown and LaTeX
# 点击一下输入框, 或者按键盘上的return回车
# ![](./img/cocalc07.png)
# 
# 然后就可以输入一段说明文字了, 与word等所见即所得的编辑器不同, 在markdown格式下, 文字的格式是以一些符号来标记的, 这样你可以专注于写作内容, 而不是排版. 排版的过程将由markdown编辑器自动完成渲染. 
# ![](./img/cocalc08.png)
# 
# 其中你还可以使用两个美元符号来标记说要写一段公式了. 数学公式也是用纯文本来编写的, 这样更容易复制粘贴和修改. 
# * 关于Mardown的格式, 请参考https://help.github.com/articles/basic-writing-and-formatting-syntax/
# * 关于撰写数学公式的LaTeX, 一开始可能会有困难, 可以使用类似 http://www.tutorialspoint.com/latex_equation_editor.htm 的在线公式编辑器, 写几次以后就能掌握常用的格式了, 比如上标就是^ 符号, 下标就是 _ 符号
# 
# 输入完一段以后, 注意使用Shift+return来"运行", 一旦运行, 刚才写的纯文本就变成排版好的文字了. 
# 
# ![](./img/cocalc09.png)
# 

# ## 代码与运行
# 
# 每个输入框默认是在code的模式下, 
# ![](./img/cocalc10.png)
# 在code的模式, 在输入框里就可以写入python代码了. 学任何一个语言, 第一个程序往往都是Hello World, 就是要让程序在屏幕上输出Hello World这个字符串. 这是程序员们的习惯, 就像刷手一样. 
# 
# python的hello world很简单, 只需要
# ```python
# print('Hello World!')
# ```
# 然后shift+return运行就可以了
# ![](./img/cocalc11.png)
# 
# 一段输入和它的输出, 整体叫做一个cell, 这是可以整体移动/复制/删除/新建等等的操作, 可以看看菜单栏中cell那一部分. 也会有快捷键, 比如进入编辑状态就是return, 退出编辑状态是esc, 按m就切换到了markdown模式, 按b是在当前cell下面插入一个新的cell, 按两下d就是删除当前cell. 
# 
# jupyter是一个一个cell按照顺序依次运行的. 
# 

# ## Kernel
# 
# Jupyter只是一个界面, 它运行的内核Kernel可以是很多种, 这个教程使用的kernel是Python3(anaconda), 请注意这一点
# ![](./img/cocalc12.png)
# 
# 写程序都会有死机的可能, 自然也需要"重启", 这个功能也在Kernel菜单里面. 
# ![](./img/cocalc13.png)
# 
# 要是人类能够轻易重启, 那该少了多少疾病啊. 

# ## 帮助
# 本文只是一个简单的入门讲解, 关于Cocalc还有很多强大的功能
# 
# cocalc的帮助系统很好, 在help菜单里有对jupyter notebook, markdown的说明
# ![](./img/cocalc14.png)
# 当然都是英文的. ** 英文差在临床医学是不可饶恕的恶行. ** 当然各种翻译网站也是越来越好的, 比如国内可以用http://fanyi.sogou.com/ deep learning的奇迹越来越多, 其中相当一部分就是用python写的呢. 
# 
# 关于Cocalc的使用, 还可以有参考文档和视频教程. 
# https://github.com/sagemathinc/cocalc/wiki/Portal
# ![](./img/cocalc15.png)
# 
# 如果这些里面都找不到你面临问题的答案, 那就给他们发support ticket邮件吧. 如果你面对的问题是大家都可能出现的, 他们甚至直接在后台升级系统. 
# 

# # 文学编程
# 
# 使用Jupyter写程序, 可以把说明部分用markdown来写, 把code写在说明后面, 然后直接附上输出结果. 这种方式就像是在写论文, 其实这种方式也有一个响亮的名字叫做"文学编程"
# 
# ```
# 文学编程（英语：Literate programming）是由高德纳提出的编程方法，希望能用来取代结构化编程范型。[1]
# 正如高德纳所构想的那样，文学编程范型不同于传统的由计算机强加的编写程序的方式和顺序，而代之以让程序员用他们自己思维内在的逻辑和流程所要求的顺序开发程序。[2]文学编程自由地表达逻辑，而且它用人类日常使用的语言写出来，就好像一篇文章一样，文章里包括用来隐藏抽象的宏和传统的源代码。
# 
# https://zh.wikipedia.org/zh-hans/%E6%96%87%E5%AD%A6%E7%BC%96%E7%A8%8B
# 
# ```
# 
# 对于编写程序的初学者, 特别是对于眼科的初学者, 我强烈建议使用文学式编程来写程序. 
# 
# 因为显然各位的临床工作都很繁忙, 自己写点东西很可能是利用下夜班或者挤出来的闲暇时间, 相信我, 你绝对不会想得起来上一次写了一半的代码是什么思路的, 等你慢慢回忆起上次精妙的算法时, 恐怕又到了出门诊或者上手术的时间了. 
# 
# 所以, 仔细用文字记录下你思路, 就像自己对自己说话一样把想法用markdown写下来, 写清楚以后再去写一小段代码, 不要写太长, 每一段代码只做很小的一件事情, 然后就运行输出这一段代码, 看看它是否能够输出你所期望的结果. 经常我们还要进行一些测试, 来"证明"写下代码的正确性. 
# 
# 几个关键经验: 
# * 说明要比代码长
# * 每段代码只做一件小事
# * 每段代码的正确性都是要"可证明"的
# 

# # 目录
# * [第 0 课, 动机](./lesson_00_motivation.html)
# * [第 1 课, CoCalc](./lesson_01_jupyter.html)
# * [第 2 课, SRK公式](./lesson_02_SRK.html)
#   * [第 2 课, jupyter笔记本下载](./lesson_02_SRK.ipynb)
# * [第 3 课, SRK-II公式](./lesson_03_SRKII.html)
#   * [第 3 课, jupyter笔记本下载](./lesson_03_SRKII.ipynb)
# * [第 4 课, 一千零一个病人](./lesson_04_1001patients.html)
#   * [第 4 课, jupyter笔记本下载](./lesson_04_1001patients.ipynb)
# * [第 5 课, save 和 load](./lesson_05_pandas.html)
#   * [第 5 课, jupyter笔记本下载](./lesson_05_pandas.ipynb)
# 

# In[ ]:




