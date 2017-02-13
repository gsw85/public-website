---
title: "深度学习框架比较：Deeplearning4j、Torch、Theano、Caffe、TensorFlow、MxNet和CNTK"
layout: cn-default
---

# 深度学习框架比较：Deeplearning4j、Torch、Theano、TensorFlow、Caffe、Paddle、MxNet、Keras和CNTK

Deeplearning4j不是第一个开源的深度学习项目，但与此前的其他项目相比，DL4J在编程语言和宗旨两方面都独具特色。DL4J是基于JVM、聚焦行业应用且提供商业支持的**分布式深度学习框架**，其宗旨是在合理的时间内解决各类涉及大量数据的问题。它与Hadoop和[Spark](./spark)集成，可使用任意数量的[GPU](./gpu)或[CPU](./native)运行，而且发生任何问题都可以[联系服务热线](http://www.skymind.io/contact)。DL4J是一种适用于各类平台的便携式学习库，并未同AWS、Azure或谷歌云等任何云端服务相捆绑。论速度，DL4J用多GPU运行非平凡图像处理任务时的[性能可媲美Caffe](https://github.com/deeplearning4j/dl4j-benchmark)，优于TensorFlow和Torch。如需对Deeplearning4j运行对标，请参阅[此页](https://deeplearning4j.org/benchmark)的指南，通过调整JVM的堆空间、垃圾回收算法、内存管理以及DL4J的ETL数据加工管道来优化DL4J的性能。 

<p align="center">
<a href="zh-quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event’, 'quickstart', 'click');">DEEPLEARNING4J快速入门指南</a>
</p>

### 目录

Lua

* <a href="#torch">Torch</a>

Python框架

* <a href="#theano">Theano及其生态系统</a>
* <a href="#tensorflow">TensorFlow</a>
* <a href="#caffe">Caffe</a>
* <a href="#cntk">CNTK</a>
* <a href="#dsstne">DSSTNE</a>
* <a href="#keras">Keras</a>
* <a href=”#mxnet">Mxnet</a>
* <a href="#paddle">Paddle</a>
* <a href="#bigdl">BigDL</a>
* <a href="#licensing">许可</a>

JVM相关因素

* <a href="#speed">速度</a>
* <a href="#java">DL4J：为什么选择JVM？</a>
* <a href="#ecosystem">DL4J：生态系统</a>
* <a href="#scala">DL4S：基于Scala语言的深度学习</a>
* <a href="#ml">机器学习框架</a>
* <a href="#tutorial">扩展阅读</a>

## Lua

### <a name="torch">Torch</a>

[**Torch**](http://torch.ch/)是用Lua编写的带API的计算框架，支持机器学习算法。Facebook和Twitter等大型科技公司使用Torch的某些版本，由内部团队专门负责定制自己的深度学习平台。Lua是上世纪九十年代早期在巴西开发的多范例脚本语言。 

Torch7虽然功能强大，[但其设计并不适合在两个群体中大范围普及](https://news.ycombinator.com/item?id=7929216)，即主要依赖Python的学术界，以及普遍使用Java的企业软件工程师。Deeplearning4j用Java编写，反映了我们对行业应用和使用便利的重视。我们认为可用性是阻碍深度学习实施工具广泛普及的限制因素。我们认为可扩展性应当通过Hadoop和Spark这样的开源分布式运行时系统来实现自动化。我们还认为，从确保工具正常运作和构建社区两方面来看，提供商业支持的开源框架是最恰当的解决方案。

Facebook于2017年1月开放了Torch的Python API――[PyTorch](https://github.com/pytorch/pytorch)的源代码。PyTorch支持动态计算图，让您能处理长度可变的输入和输出，而这在RNN应用和其他一些情形中很有帮助。CMU的DyNet和PFN的Chainer框架也支持动态计算图。 

利与弊：

* (+) 大量模块化组件，容易组合
* (+) 很容易编写自己的层类型并在GPU上运行
* (+) Lua.;) （大多数学习库的代码是Lua，比较易读）
* (+) 有很多已预定型的模型
* (+) PyTorch
* (-) Lua
* (-) 通常需要自己编写定型代码（即插即用相对较少）
* (-) 不提供商业支持
* (-) 文档质量参差不齐

## Python框架

### <a name="theano">Theano及其生态系统</a>

深度学习领域的许多学术研究者依赖[**Theano**](http://deeplearning.net/software/theano/)，Theano是深度学习框架中的元老，用[Python](http://darkf.github.io/posts/problems-i-have-with-python.html)编写。Theano和NumPy一样，是处理多维数组的学习库。Theano可与其他学习库配合使用，非常适合数据探索和研究活动。 

现在已有大量基于Theano的开源深度学习库，包括[Keras](https://github.com/fchollet/keras)、 [Lasagne](https://lasagne.readthedocs.org/en/latest/)和[Blocks](https://github.com/mila-udem/blocks)。这些学习库试着在Theano有时不够直观的界面之上添加一层便于使用的API。（截至2016年3月，另一个与Theano相关的学习库[Pylearn2似乎已经停止开发](https://github.com/lisa-lab/pylearn2)。）

相比之下，Deeplearning4j致力于将深度学习引入生产环境，以Java和Scala等JVM语言开发解决方案。力求以可扩展、多个GPU或CPU并行的方式让尽可能多的控制点实现自动化，在需要时与Hadoop和[Spark](./spark.html)集成。 

利与弊

* (+) Python + NumPy
* (+) 计算图是良好的抽象化方式
* (+) RNN与计算图匹配良好
* (-) 原始的Theano级别偏低
* (+) 高级的包装界面（Keras、Lasagne）减少了使用时的麻烦
* (-) 错误信息可能没有帮助
* (-) 大型模型的编译时间可能较长
* (-) 比Torch笨重许多
* (-) 对已预定型模型的支持不够完善
* (-) 在AWS上容易出现bug


### <a name="tensorflow">TensorFlow</a>

* 谷歌开发了TensorFlow来取代Theano，这两个学习库其实很相似。有一些Theano的开发者在谷歌继续参与了TensorFlow的开发，其中包括后来加入了OpenAI的Ian Goodfellow。 
* 目前**TensorFlow**还不支持所谓的“内联（inline）”矩阵运算，必须要复制矩阵才能对其进行运算。复制非常大的矩阵会导致成本全面偏高。TF运行所需的时间是最新深度学习工具的四倍。谷歌表示正在解决这一问题。 
* 和大多数深度学习框架一样，TensorFlow是用一个Python API编写的，通过C/C++引擎加速。这种解决方案并不适合Java和Scala用户群。 
* TensorFlow的用途不止于深度学习。TensorFlow其实还有支持强化学习和其他算法的工具。
* 谷歌似乎已承认TensorFlow的目标包括招募人才，让其研究者的代码可以共享，推动软件工程师以标准化方式应用深度学习，同时为谷歌云端服务带来更多业务――TensorFlow正是为该服务而优化的。 
* TensorFlow不提供商业支持，而谷歌也不太可能会从事支持开源企业软件的业务。谷歌的角色是为研究者提供一种新工具。 
* 和Theano一样，TensforFlow会生成计算图（如一系列矩阵运算，例如z = sigmoid(x)，其中x和z均为矩阵），自动求导。自动求导很重要，否则每尝试一种新的神经网络设计就要手动编写新的反向传播算法，没人愿意这样做。在谷歌的生态系统中，这些计算图会被谷歌大脑用于高强度计算，但谷歌还没有开放相关工具的源代码。TensorFlow可以算是谷歌内部深度学习解决方案的一半。 
* 从企业的角度看，许多公司需要思考的问题在于是否要依靠谷歌来提供这些工具。 
* 注意：有部分运算在TensorFlow中的运作方式与在NumPy中不同。 

利与弊

* (+) Python + NumPy
* (+) 与Theano类似的计算图抽象化
* (+) 编译时间快于Theano
* (+) 用TensorBoard进行可视化
* (+) 同时支持数据并行和模型并行
* (-) 速度比其他框架慢
* (-) 比Torch笨重许多；更难理解
* (-) 已预定型的模型不多
* (-) 计算图纯粹基于Python，因此速度较慢
* (-) 不提供商业支持
* (-) 加载每个新的定型批次时都要跳至Python
* (-) 不太易于工具化
* (-) 动态类型在大型软件项目中容易出错

### <a name="caffe">Caffe</a>

[**Caffe**](http://caffe.berkeleyvision.org/)是一个广为人知、广泛应用的机器视觉库，将Matlab实现的快速卷积网络移植到了C和C++平台上（[参见Steve Yegge关于一个芯片一个芯片地移植C++代码的博客，可以帮助您思考如何在速度和这种特定的技术债务之间进行权衡](https://sites.google.com/site/steveyegge2/google-at-delphi)）。Caffe不适用于文本、声音或时间序列数据等其他类型的深度学习应用。与本文提到的其他一些框架相同，Caffe选择了Python作为其API。 

Deeplearning4j和Caffe都可以用卷积网络进行图像分类，这是最先进的技术。与Caffe不同，Deeplearning4j*支持*任意芯片数的GPU并行运行，并且提供许多看似微不足道，却能使深度学习在多个并行GPU集群上运行得更流畅的功能。虽然在论文中被广泛引述，但Caffe主要用于为其Model Zoo网站提供已预定型的模型。Deeplearning4j正在开发将Caffe模型导入Spark的[开发解析器](https://github.com/deeplearning4j/deeplearning4j/pull/480)。

利与弊：

* (+) 适合前馈网络和图像处理
* (+) 适合微调已有的网络
* (+) 定型模型而无需编写任何代码
* (+) Python接口相当有用
* (-) 需要用C++ / CUDA编写新的GPU层
* (-) 不适合循环网络
* (-) 用于大型网络（GoogLeNet、ResNet）时过于繁琐
* (-) 不可扩展，有些不够精简
* (-) 不提供商业支持

### <a name="cntk">CNTK</a>

[**CNTK**](https://github.com/Microsoft/CNTK)是微软的开源深度学习框架。CNTK的全称是“计算网络工具包。”此学习库包括前馈DNN、卷积网络和循环网络。CNTK提供基于C++代码的Python API。虽然CNTK遵循一个[比较宽松的许可协议](https://github.com/Microsoft/CNTK/blob/master/LICENSE.md)，却并未采用ASF 2.0、BSD或MIT等一些较为传统的许可协议。这一许可协议不适用于CNTK旨在简化分布式定型的1-Bit随机梯度下降（SGD）方法，该组件不得用作商业用途。 

### <a name="dsstne">DSSTNE</a>

亚马逊的深度可伸缩稀疏张量网络引擎又称[DSSTNE](https://github.com/amznlabs/amazon-dsstne)，是用于机器学习和深度学习建模的学习库。它是在TensorFlow和CNTK之后较晚发布的不少开源深度学习库之一，后来亚马逊开始以AWS力挺MxNet，所以DSSTNE的发展前景不太明朗。DSSTNE主要用C++写成，速度较快，不过吸引到的用户群体规模尚不及其他学习库。 

* (+) 可实现稀疏编码
* (-) 亚马逊可能没有共享[示例运行结果最优化所必需的全部信息](https://github.com/amznlabs/amazon-dsstne/issues/24)
* (-) 亚马逊已决定在AWS上使用另一个框架。

### <a name="keras">Keras</a>

[Keras](keras.io)是一个基于Theano和TensorFlow的深度学习库，具有一个受Torch启发、较为直观的API。这可能是目前最好的Python API。Deeplearning4j[可以导入Keras模型](./keras)。Keras是由谷歌软件工程师[Francois Chollet](https://twitter.com/fchollet)开发的。 

### <a name="mxnet">MxNet</a>

[MxNet](https://github.com/dmlc/mxnet)是一个提供多种API的机器学习框架，主要面向R、Python和Julia等语言，目前已被[亚马逊云服务](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html)采用。据传Apple在2016收购Graphlab/Dato/Turi后，有些部门也开始使用MxNet。MxNet是一个快速灵活的学习库，由华盛顿大学的Pedro Domingos及其研究团队运作。MxNet与Deeplearning4j某些方面的比较参见[此处](https://deeplearning4j.org/mxnet)。 

### <a name="paddle">Paddle</a>

[Paddle](https://github.com/PaddlePaddle/Paddle)是[由百度开发支持的深度学习框架](http://www.infoworld.com/article/3114175/artificial-intelligence/baidu-open-sources-python-driven-machine-learning-framework.html)，全名为PArallel Distributed Deep LEarning，即
“并行分布式深度学习”。Paddle是最新发布的大型深度学习框架；与多数框架一样，它提供一个Python API。 

### <a name="bigdl">BigDL</a>

[BigDL](https://github.com/intel-analytics/BigDL)是一个新的深度学习框架，重点面向Apache Spark且只能在英特尔处理器上运行。 

### <a name="licensing">许可</a>

上述开源项目的另一区别在于其许可协议：Theano、Torch和Caffe采用BSD许可协议，未能解决专利和专利争端问题。Deeplearning4j和ND4J采用**[Apache 2.0许可协议](http://en.swpat.org/wiki/Patent_clauses_in_software_licences#Apache_License_2.0)**发布。该协议包含专利授权和防止报复性诉讼的条款，也就是说，任何人都可以自由使用遵循Apache 2.0协议的代码创作衍生作品并为其申请专利，但如果对他人提起针对原始代码（此处即DL4J）的专利权诉讼，就会立即丧失对代码的一切专利权。（换言之，这帮助你在诉讼中进行自我防卫，同时阻止你攻击他人。）BSD一般不能解决这个问题。 

## JVM相关因素

### <a name="speed">速度</a>

Deeplearning4j依靠ND4J进行基础的线性代数运算，事实表明其处理大矩阵乘法的[速度至少是NumPy的两倍](http://nd4j.org/benchmarking)。这正是DL4J被NASA的喷气推进实验室所采用的原因之一。此外，Deeplearning4j为多芯片运行而优化，支持采用CUDA C的x86和GPU。

虽然Torch7和DL4J都采用并行运行，DL4J的**并行运行是自动化的**。我们实现了从节点（worker nodes）和连接的自动化设置，让用户在[Spark](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark)、[Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn)或[Akka和AWS](http://deeplearning4j.org/scaleout.html)环境中建立大型并行网络时可以绕过学习库。Deeplearning4j最适合快速解决具体问题。 

Deeplearning4j的所有功能参见[功能介绍](./features.html)。

### <a name="java">为什么选择JVM？</a>

经常有人问，既然有如此之多的深度学习用户都专注于Python，为什么我们要开发一个面向JVM的开源深度学习项目。的确，Python有着优越的语法要素，可以直接将矩阵相加，而无需像Java那样先创建显式类。Python还有由Theano、NumPy等原生扩展组成的广泛的科学计算环境。

但JVM及其主要语言Java和Scala仍然具备几项优势。 

首先，多数大型企业和政府机构高度依赖Java或基于JVM的系统。他们已经进行了巨大的投资，而基于JVM的人工智能可以帮助他们充分实现这些投资的价值。在企业界，Java依然是应用范围最广的语言。Java是Hadoop、ElasticSearch、Hive、Lucene和Pig的语言，而它们恰好都是解决机器学习问题的有用工具。Spark和Kafka则是用另一种JVM语言Scala编写的。也就是说，深度学习本可以帮助许多需要解决现实问题的程序员，但他们却被语言屏障阻碍。我们希望提高深度学习对于这一广大群体的可用性，这些新的用户可以将深度学习直接付诸实用。Java是世界上用户规模最大的编程语言，共有1000万名开发者使用Java。 

其次，与Python相比，Java和Scala在速度上具有先天优势。如不考虑依赖用Cython加速的情况，任何用Python写成的代码在根本上速度都相对较慢。不可否认，运算量最大的运算都是用C或C++语言编写的。（此处所说的运算也包括高级机器学习流程中涉及的字符和其他任务。）大多数最初用Python编写的深度学习项目在用于生产时都必须重新编写。Deeplearning4j依靠[JavaCPP](https://github.com/bytedeco/javacpp)从Java中调用预编译的原生C++代码，大幅提升定型速度。许多Python程序员选择用Scala实现深度学习，因为他们在进行共享基本代码的合作时更青睐静态类型和函数式编程。 

第三，为了解决Java缺少强大的科学计算库的问题，我们编写了[ND4J](http://nd4j.org)。ND4J在分布式CPU或GPU上运行，可以通过Java或Scala的API进行对接。

最后，Java是一种安全的网络语言，本质上具有跨平台的特点，可在Linux服务器、Windows和OSX桌面、安卓手机上运行，还可通过嵌入式Java在物联网的低内存传感器上运行。Torch和Pylearn2通过C++进行优化，优化和维护因而存在困难，而Java则是“一次编写，随处运行”的语言，适合需要在多个平台上使用深度学习系统的企业。 

### <a name="ecosystem">生态系统</a>

生态系统也是为Java增添人气的优势之一。[Hadoop](https://hadoop.apache.org/)是用Java实施的；[Spark](https://spark.apache.org/)在Hadoop的Yarn运行时中运行；[Akka](https://www.typesafe.com/community/core-projects/akka)等开发库让我们能够为Deeplearning4j开发分布式系统。总之，对几乎所有应用而言，Java的基础架构都经过反复测试，用Java编写的深度学习网络可以靠近数据，方便广大程序员的工作。Deeplearning4j可以作为YARN的应用来运行和预配。

Scala、Clojure、Python和Ruby等其他通行的语言也可以原生支持Java。我们选择Java，也是为了尽可能多地覆盖主要的程序员群体。 

虽然Java的速度不及C和C++，但它仍比许多人想象得要快，而我们建立的分布式系统可以通过增加节点来提升速度，节点可以是GPU或者CPU。也就是说，如果要速度快，多加几盒处理器就好了。 

最后，我们也在用Java为DL4J打造NumPy的基本应用，其中包括ND-Array。我们相信Java的许多缺点都能很快克服，而其优势则大多会长期保持。 

### <a name="scala">Scala</a>

我们在打造Deeplearning4j和ND4J的过程中特别关注[Scala](./scala)，因为我们认为Scala具有成为数据科学主导语言的潜力。用[Scala API](http://nd4j.org/scala.html)为JVM编写数值运算、向量化和深度学习库可以帮助整个群体向实现这一目标迈进。 

关于DL4J与其他框架的不同之处，也许只需要[尝试一下](./quickstart)就能有深入的体会。

### <a name="ml">机器学习框架</a>

上文提到的深度学习框架都是比较专业化的框架，此外还有许多通用型的机器学习框架。这里列举主要的几种：

* [sci-kit learn](http://scikit-learn.org/stable/)－Python的默认开源机器学习框架。 
* [Apache Mahout](https://mahout.apache.org/users/basics/quickstart.html)－Apache的主打机器学习框架。Mahout可实现分类、聚类和推荐。
* [SystemML](https://sparktc.github.io/systemml/quick-start-guide.html)－IBM的机器学习框架，可进行描述性统计、分类、聚类、回归、矩阵参数化和生存分析，还包括支持向量机。 
* [微软DMTK](http://www.dmtk.io/)－微软的分布式机器学习工具包。分布式词嵌入和LDA。 

### <a name="tutorial">Deeplearning4j教程</a>

* [深度神经网络简介](./neuralnet-overview)
* [卷积网络教程](./zh-convolutionalnets)
* [LSTM和循环网络教程](./zh-lstm)
* [通过DL4J使用循环网络](./zh-usingrnns)
* [MNIST中的深度置信网络](./deepbeliefnetwork)
* [用Canova定制数据加工管道](./zh-image-data-pipeline)
* [受限玻尔兹曼机](./zh-restrictedboltzmannmachine)
* [本征向量、PCA和熵](./zh-eigenvector)
* [深度学习词汇表](./glossary.html)
* [Word2vec、Doc2vec和GloVe](./zh-word2vec)
