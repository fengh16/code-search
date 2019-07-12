# 1 问题

## 1.1 A Survey of Machine Learning for Big Code and Naturalness

### 1.1.1 笔记

自然假设：软件是人类沟通的方式；软件语料库和自然语言语料库有类似的统计学性质；这些性质可以被用于构建更好的软件工程工具。

源码有两个audience，它是bimodal，既和人类沟通（被人类理解）又和电脑沟通（被电脑执行）。

代码概率模型可以被用于：
- **代码生成**：给定数据集$\mathfrak{D}$、代码表示$c$，可为空的上下文$C(c)$，学习出概率分布$P_\mathfrak{D}(c|C(c))$。
  - 如果$C(c) = \emptyset$，则$P_\mathfrak{D}$是代码的**语言模型**；
  - 当$C(c)$是非代码形式，则$P_\mathfrak{D}$是代码的**代码生产多模式模型**；
  - 当$C(c)$是代码，则$P_\mathfrak{D}$是代码的**传感器模型**。除了生成代码，$P_\mathfrak{D}$也可以作为代码自然程度的衡量函数。

  在代码生成模型中表示代码有如下3种：
  - **token层（序列）**，$c=t_1\cdots t_M$，直接一步能预测序列是不现实的，所以一般采用单个元素预测，模型为$p(t_m|t_1\cdots t_{m-1},C(c))$；
    - **n-gram模型**：假定代码生成从左至右，只使用前$n-1$个token生成。有这个公式，$P_{\mathfrak{D}}(c|C(c))=P(t_1\cdots t_M|C(c))=\prod\limits_{m-1}^MP(t_m|t_{m-1}\cdots t_{m-n+1},C(c))$。Nguyen为每个token加上了解析信息。Tu et al等人发现代码有很强的局部性，因而增加了缓存机制，对刚见过的token设置更高的优先级。
    - **RNN&LSTM神经网络**：性能更好，但训练时间非常长。
  - **语法层（树）**；从顶向下，从左向右生成抽象语法树。
  - **语义层（图）**

  代码生成模型的种类，依照$C(c)$的划分有3类：
  - **语言模型**：$C(c)=\emptyset$，使用交叉熵$H(c,P_\mathfrak{D})=-\frac{1}{M}log_2P_\mathfrak{D}(c)$。
  - **代码传感器模型**：例如翻译编程语言，假定源代码与目标代码的短语有映射关系。
  - **多模式模型**：例如从代码、规格或者查询中生成代码。
- **代码表示**：建模为$P_\mathfrak{D}(\pi|f(c))$，其中$f$是$c$到目标表示的转换，$\pi$是任意特征。有两种类型，他们不是互斥的，可以结合。
  - **分布表示**：指将特征分布到一个向量或矩阵。
  - **结构预测**：指预测的是结构化对象。
- **模式挖掘**：非监督学习学习模式，建模为$P_\mathfrak{D}(f(c))=\sum\limits_IP_\mathfrak{D}(g(c)|I)P(I)$，其中$g$返回代码的视图，I是模型引入的潜在变量。

代码概率模型有如下应用：
- **推荐系统**：如自动补全
- **代码风格推断**
- **寻找代码缺陷**
- **代码翻译**
- **代码与文本互相转换**
- **文档、追溯和信息提取**
- **程序合成**
- **程序分析**

### 1.1.2 问题

- 自然假设是否成立？形式语言与自然语言有较大的不同，相较于自然语言，它更准确，无二义性，容错率更低，可以有嵌套递归的复杂语法，词素则很少，很少有成俗的语法特例。
- 没怎么看懂模式挖掘的建模公式。

## 1.2 code2vec: Learning Distributed Representations of Code

### 1.2.1 笔记

主要的想法是将代码片段表示成一个固定长度的**代码向量**，来预测其语义性质。

以$C$代表代码段，$L$代表类标，潜在假设类标分布可以从$C$的语法路径推断出来。模型试图学习$P(L|C)$。这还可以用于捕捉名字之间的相似程度。

应用有：
- 代码审核：推荐更好的方法名
- 提取并发现API

有如下的挑战：
- 表示：使用AST的路径作为输入而不是token。
- 注意力：学习哪些路径更重要。

AST可以表示为5元组$\langle N,T,C,s,\delta,\phi\rangle$，其中$N$是非终结符集，$T$是终结符集，$s$是根节点，$\delta:N\to (N\cup T)^*$是非终结符到子节点的映射，$\phi:T\to X$是终结符到值的映射。长度为$k$的路径可表示为序列$n_1d_1\cdots n_kd_kn_{k+1}$，其中$n_1,n_{k+1}\in T$，$\forall i\in [2..k]:n_i\in N$，$\forall i \in [1..k]: d_i\in\{\uparrow, \downarrow\}$表示路径移动方向。$start(p)=n_1$，$end(p)=k+1$。路径上下文是三元组$\langle x_s,p,x_t\rangle$，其中$x_s=\phi(start(p))$，$x_t=\phi(end(p))$。

$Rep$表示一段代码。$TPairs(C)=\{(term_i, term_j|term_i,term_j\in termNodes(C)\land i\neq j\}$，$Rep(C)=\{(x_s,p,x_t)|\exists(term_s,term_t)\in TPairs(C):x_s=\phi(term_s)\land x_t=\phi(term_t)\land start(p)=term_s\land end(p)=term_t\}$。

路径注意力模型，使用以下的组件：路径和名字的嵌入（$path\_vocab\in\mathbb{R}^{|P|\times d},value\_vocab\in\mathbb{R}^{|X|\times d}$，其中$P$是AST路径的集合，$X$是AST终结符的值的集合）、全连层（矩阵$W$）、注意力向量（$\boldsymbol{a}$）和标签的嵌入($tags\_vocab$)。路径和名字的嵌入矩阵随机初始化。$W$的宽度是嵌入的大小$d$，这是经验确定的超参数，一般为100-500。上下文向量$\boldsymbol{c}_i=embedding(\langle x_s,p_j,x_t\rangle)=[value\_vocab_s;path\_vocab_j;value\_vocab_t]\in\mathbf{R}^{3d}$。$\tilde{\boldsymbol{c}}_i=\tanh(W\cdot\boldsymbol{c}_i)$是全连层的输出，称为组合上下文向量，其中$W\in\mathrm{R}^{d\times3d}$。注意力向量$\boldsymbol{a}\in\mathrm{R}^d$随机初始化。注意力权重$\alpha_j=\frac{\exp(\tilde{\boldsymbol{c}}_i^T\cdot\boldsymbol{a})}{\sum_{j=1}^n\exp(\tilde{\boldsymbol{c}}_j^T\cdot\boldsymbol{a})}$。代码向量$\boldsymbol{v}=\sum\limits_{i=1}^{n}a_i\cdot\tilde{\boldsymbol{c}}_i$。进行预测时，定义需要学习的参数$tags\_vocab\in\mathbb{R}^{|Y|\times d}$，$Y$是标签集，分布$q(y)$这样表示$q(y_i)=\frac{\exp(\boldsymbol{v}^T\cdot tags\_vocab_i)}{\sum_{y_j\in Y}\exp(\boldsymbol{v}^T\cdot tags\_vocab_j)}$。

训练时，真值采用one-hot向量，loss函数是交叉熵，即负对数，$-\sum\limits_{y\in Y}p(y)\log q(y)$，其中p是真实分布。

### 1.2.2 问题

- 在注意力模型那里，一开始提到全连层嵌入函数的宽度是$d$，后来宽度是$3d$，这是笔误么？

## 1.3 Maybe Deep Neural Networks are the Best Choice for Modeling

### 1.3.1 笔记

使用单层GRU。分割token到subword，使用`</t>`代表token边界。对一个token内出现频率较高的subword对，执行合并操作。

在从subword预测完整token的方法中（Beam Search），`predict`函数返回在当前subword序列下的候选subword及它们的可能性。使用了两个优先级队列，一个是`candidates`，对仍需探索的subword序列排名，另一个是`bestTokens`，包含目前找到的前k个最有可能的完整token。主循环从`candidates`中pop出前$b$个最好的，尝试扩展它们，并测试这些扩展的可能性。如果扩展是以`</t>`结尾的，则将他们添加到`bestTokens`中，否则添加到`candidates`中。直到遇到一些条件循环终止。

### 1.3.2 问题

- 单层GRU是否足够解决问题，有没有考虑多层？

## 1.4 How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning

### 1.4.1 笔记

只觉得的想法是将文字和代码通过两个encoder都映射到同一个向量空间中，再使用`cos`距离判断相似程度。我们使用预训练模型，再微调它。

分以下5个步骤：

1. 获取并解析数据：Google将GitHub的开源数据存放在[BigQuery](https://cloud.google.com/bigquery/)。获取数据后，文件被解析成代码和文档对，其中代码被移除了注释（Python可以使用标注库`ast`完成）。而后将数据分为训练、验证和测试。
2. 使用Seq2Seq建立一个代码摘要。
3. 训练一个编码自然语言短语的语言模型。
4. 训练从代码向量映射到与自然语言相同的向量的模型。
5. 创建一个语义搜索工具。

### 1.4.2 问题

- 这篇文章讲解得非常有条理。要有问题的话，就是Seq2Seq和语言模型具体的实现是什么？

## 1.5 Deep API Learning

### 1.5.1 笔记

大致是将用户的输入作为源语言，API序列作为目标语言，使用基于attention的RNN Encoder-Decoder模型。

使用基于IDF的权重来处理API重要程度不同的问题。$w_{idf}(y_t)=\log(\frac{N}{n_{y_t}})$，其中$N$是API序列总数，$n_{y_t}$是$y_t$出现的文件数目。新的损失函数即为$\mathrm{cost}_{it}=-\log p_\theta(y_{it}|x_i)-\lambda w_{idf}(y_t)$。

数据集是从GitHub上获取的$\langle \text{API sequence},\text{annotation}\rangle$对。抽取API sequence时，遍历AST，将`new C()`转化为`C.new`，将`o.m()`（`o`是`C`的实例）转化为`C.m`。AST的遍历为后序。抽取annotation时，选择方法注释的第一句话作为总结。

训练时，使用GRU变种作为RNN的结构。encoder使用了两个RNN，一个forward RNN编码源语句，一个backward RNN编码逆源语句。decoder也是一个RNN。所有的RNN都有1000个隐藏单元。单词嵌入的大小为120。使用Adadelta自动修改学习率。

使用Beam Search的方法启发式地选择序列。

### 1.5.2 问题

- Encoder为什么使用两个RNN？没看懂它们的用途。

## 1.6 Deep Code Search

### 1.6.1 笔记

CODEnn将代码和自然语言映射到同一向量空间（Joint embedding）。在做序列嵌入的时候，使用了RNN，RNN的所有输出再经过最大池化。整个网络由如下3部分组成：
- 代码嵌入网络（CoNN）：考虑3方面，方法名、API调用序列和token。方法名和API调用序列使用RNN，token使用多层感知机，所有的结果都经过最大池化。最后在经过单层全连层映射到向量空间，激活函数使用$\tanh$。
- 描述嵌入网络（DeNN）：使用RNN和最大池化。
- 相似度模块：使用cos距离。

loss采用$L(\theta)=\sum\limits_{\langle C,D^+,D^-\rangle\in P}\max(0,\epsilon-\cos(\boldsymbol{c},\boldsymbol{d}^+)+\cos(\boldsymbol{c},\boldsymbol{d}^-))$。其中$\theta$是参数，$P$是数据集，$\boldsymbol{c},\boldsymbol{d}^+,\boldsymbol{d}^-$分别是代码向量，正例向量和反例向量。

### 1.6.2 问题

- token信息为什么不采用序列，使用RNN训练而是无序的，使用MLP训练？

## 1.7 Aroma: Code Recommendation via Structural Code Search

### 1.7.1 笔记

AROMA是代码到代码的查询。它首先会粗略地从给定语料库搜索一小部分（约1000），而后再对得到的代码片段进行修剪使之与原来的代码更相似，再根据相似度排序，再进行聚类，最后再对聚类取交。

轻量搜索时，先将代码的树提取特征，与代码库的特征进行点乘，再排序。

引入关键字token（如`while`、`{`）、非关键字token（标识符和字面常量）、简化解析树的概念。

### 1.7.2 问题

# 2 想法
