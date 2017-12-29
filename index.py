
# coding: utf-8

# # 面向眼科医生的Python教程
# 
# # 序言
# 
# 
# 本教程是面向眼科医生的python语言教程。python语言很简单, 难在开始学习和坚持学习前三个月。
# 
# 人之为学有难易乎? 靡不有初鲜克有终。
# 
# 编程是一种技能，学习编程和做手术一样，需要由易到难，勤于练习。不要寄希望于看一遍教程或者书籍就掌握写python的技巧，也不必强求自己一开始就写出华丽复杂的程序。
# 
# 本教程以眼科临床工作和科研中常见的场景循序渐进进行教学，并且在每一部分都有练习作业。请在阅读过程中随时停下来，自己输入代码试一试。多多练习才能体会到编程的乐趣。
# 
# 测试代码和完成作业推荐使用[cocalc.com](http://cocalc.com/) 这个在线编程环境。您可以将本教程中的ipython文件上传到自己的cocalc项目内，在其中修改代码并运行。
# 
# 或者, 我最近刚刚发现微软也推出了在线版本的Jupyter服务, 您也可以使用本教程在[azure notebook上的镜像](https://notebooks.azure.com/goldengrape/libraries/Python-for-ophthalmologist) . 方便一键Clone, 对于没接触过git的初学者更加方便
# 
# 当您对python足够熟悉以后再过渡到本地安装python。不要急于求成，就像应该先以翼状胬肉手术练习好显微镜使用，然后才开始phaco或玻切手术的训练。
# 
# 本教程只是一个初级的入门教程，以科学计算部分为主。在您今后的学习工作中可能还需要编写收集数据的网络爬虫、深度神经网络、甚至手机app，这些内容请参考网上其他教程。
# 
# # 使用
# 您可以使用[点击此处](https://goldengrape.github.io/Python-for-ophthalmologist)来阅读[本教程的html版本](https://goldengrape.github.io/Python-for-ophthalmologist), 这个版本应该也可以在手机上正确显示. 
# 
# 您可以点击 .ipynb 文件, 在线阅读本教程内容.
# 
# ![clickipynb](https://i.loli.net/2017/09/28/59cd07ccaf950.png)
# 
# 还可以点击 clone or download, 将本教程下载到电脑上阅读.
# 
# ![clickdownload](https://i.loli.net/2017/09/28/59cd07b75acca.png)
# 
# 下载到电脑上阅读时, 您可以打开 .html文件阅读.
# 
# 如果您阅读的是azure notebooks上的[镜像](https://notebooks.azure.com/goldengrape/libraries/Python-for-ophthalmologist), 可以一键Clone, 边看边练. 
# ![](./img/azurenotebooks.png)
# 
# # CoCalc
# 
# [Cocalc.com](https://cocalc.com/)是一个非常优秀的在线科学计算工具集, 其中已经包含了python, 以及几乎所有科学计算所需要的python工具包.
# 
# 注册cocalc是免费的, 就初学者而言, 暂时还不需要购买其中的付费升级. 主要区别是远程主机联网的能力和运行稳定性. 当您已经达到一定水平后, 比如需要使用编写网页爬虫采集数据, 可以考虑购买升级. 目前本教程使用免费版足以.
# 
# 您可以将下载下来的本教程文件upload到cocalc中, 就可以直接运行 .ipynb 文件, 并且修改其中的作业练习代码, 完成练习.
# 
# 关于coclac的使用, 本教程中将有介绍.
# 
# # Azure Notebook
# [Azure Notebook](https://notebooks.azure.com) 是微软在自己的云服务上推出的在线编程工具, 当然也包含了python, 经过测试我发现大多数的python工具包也都已经包括了, 比cocalc更好一些的是, azure notebook中即使是免费账户也可以有权限自己在安装一些工具包. 毕竟微软财大气粗. 
# 
# 在azure notebook里, 有一键Clone的工具, 您只需要点击clone按钮, 就可以原样复制本教程, 并在自己的浏览器上进行修改和运行测试. 玩坏了也没关系, 大不了删掉再重新clone一次. 
# ![](./img/azurenotebooks.png)
# 
# # 参考
# * 任何时候, Google都是您的良师益友. 凡是有确定答案的问题, 通常都可以通过Google搜索到. 至于如何能够访问Google, 已经不在本教程范围内讨论.
# 
# * [stackoverflow](https://stackoverflow.com/questions/tagged/python) 是一个编程提问网站, 多数情况下, 您遇到的问题在里面已经有所解答.
# 
# * [莫烦PYTHON教程](https://morvanzhou.github.io/tutorials/python-basic/) 是很好的视频教程, 有时候文字解释不如视频来得清晰明了.
# 
# * [廖雪峰Python教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000) 也是不错的教程, 但我推荐将其作为速查手册, 需要了解python的某个细节时去查阅其中的文字说明. 我是不推荐按照他的教程在本地装python开始学的.
# 
# * [Sololearn](https://www.sololearn.com/Course/Python/) Sololearn是个教程APP, 有android/iOS和网页版, 里面也是循序渐进的讲解了python的各个方面. 而且有实时的练习, 利用碎片时间也可以学得很好, 非常推荐.
# 
# 综上,  我推荐您:
# 1. 在手机上安装Sololearn, 利用碎片时间入门python;
# 2. 注册使用[CoCalc](https://coclac.com/app) 在线阅读本教程并做与眼科相关的练习作业.
# 3. 逐渐开始您自己的python项目, 遇到问题时查询[廖雪峰Python文字教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000), 看[莫烦PYTHON视频教程](https://morvanzhou.github.io/tutorials/python-basic/) , 并请教Google/ [stackoverflow](https://stackoverflow.com/questions/tagged/python)
# 
# # 学而时习之不亦说乎
# 祝学习愉快
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
# * [第 6 课, 统计分析](./lesson_06_pymc3-bayes.html)
#   * [第 6 课, jupyter笔记本下载](./lesson_06_pymc3-bayes.ipynb)
# 

# In[ ]:




