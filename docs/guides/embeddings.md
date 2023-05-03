# 向量化（Embeddings）

## 向量化（Embeddings）是什么？

OpenAI 的文本用向量化度量文本字符串的相关性。向量化通常用于：

- **搜索**（结果按与查询字符串相关性排名）
- **聚类**（文本字符串按相似性分组）
- **推荐**（推荐具有相关文本字符串的项目）
- **异常检测**（识别与相关性较低的异常值）
- **多样性测量**（分析相似性分布）
- **分类**（将文本字符串按其最相似的标签分类）

向量化之后是一个可浮动数字列表的向量。两个向量间的距离度量其相关性，小[距离](https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use)表明其相关性高，大距离表明其相关性低。

请访问我们的[定价页面](https://openai.com/api/pricing/)了解 Embeddings 定价。价格是基于请求的[输入](https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-input)中的 token 来计算。

:::info 提示

要查看向量化的示例，请查看我们的代码示例：

- 分类
- 主题聚类
- 搜索
- 推荐

<a tabindex="0" style="color:#fff" class="btn btn-sm btn-filled btn-primary" href="https://platform.openai.com/docs/guides/embeddings/use-cases"><span class="btn-label-wrap"><span class="btn-label-inner">Browse Samples‍</span></span></a>
:::

## 如何获得 embeddings

要获取 embeddings，请将您的文本字符串发送到向量化 API 接口，并选择向量化模型 ID（例如：`text-embedding-ada-002`）。响应将包含一个 embeddings，您可以提取、保存和使用。

示例请求：

:::code-group

```curl
curl https://api.openai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "input": "Your text string goes here",
    "model": "text-embedding-ada-002"
  }'
```

```python
response = openai.Embedding.create(
    input="Your text string goes here",
    model="text-embedding-ada-002"
)
embeddings = response['data'][0]['embedding']
```

:::

示例响应：

```
{
  "data": [
    {
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        ...
        -4.547132266452536e-05,
        -0.024047505110502243
      ],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "text-embedding-ada-002",
  "object": "list",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

请参阅[OpenAI Cookbook](https://github.com/openai/openai-cookbook/)中更多的 Python 代码示例。

使用 OpenAI Embedding 时，请注意它们的[限制和风险](https://platform.openai.com/docs/guides/embeddings/limitations-risks)。

## embedding 模型

OpenAI 提供了一个第二代 Embedding 模型（在模型 ID 中用`-002` 表示）和 16 个第一代模型（在模型 ID 中用`-001` 表示）。

我们推荐几乎所有用例都使用 `text-embedding-ada-002`。它更好、更便宜、更简单。请阅读[博客公告](https://openai.com/blog/new-and-improved-embedding-model)。

| 模型生成 | 分词器      | 最大输入标记 | 知识截止日期 |
| -------- | ----------- | ------------ | ------------ |
| V2       | cl100k_base | 8191         | 2021 年 9 月 |
| V1       | GPT-2/GPT-3 | 2046         | 2020 年 8 月 |

请参阅 OpenAI Cookbook 中更多的 Python 代码示例。

使用价格每输入 token 计价，每 1000 个 token 收费 0.0004 美元，大约是 3000 页书 1 美元（假设每页约有 800 个标记）：

| 模型                   | 每美元的粗略页面 | BEIR 搜索评估的例子性能 |
| ---------------------- | ---------------- | ----------------------- |
| text-embedding-ada-002 | 3000             | 53.9                    |
| \*-davinci-\*-001      | 6                | 52.8                    |
| \*-curie-\*-001        | 60               | 50.9                    |
| \*-babbage-\*-001      | 240              | 50.4                    |
| \*-ada-\*-001          | 300              | 49.0                    |

### 第二代模型

| 模型名称               | 分词器      | 最大输入标记 | 输出维度 |
| ---------------------- | ----------- | ------------ | -------- |
| text-embedding-ada-002 | cl100k_base | 8191         | 1536     |

:::details 第一代模型（不推荐）
[查看](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings?lang=python)
:::

## 应用场景

这里我们展示一些代表性的应用场景。我们将使用亚马逊[精细食品评论](https://www.kaggle.com/snap/amazon-fine-food-reviews)数据集进行以下示例。

### 获取 Embedding

该数据集包含截至 2012 年 10 月由亚马逊用户留下的共计 568,454 个食品评论。我们将使用最近的 1,000 条评论的子集进行演示。评论是用英语编写的，往往是积极或消极的。每条评论都有一个 ProductId、UserId、Score、评论标题（Summary）和评论正文（Text）。例如：

| 产品 ID    | 用户 ID        | 得分 | 摘要         | 正文                       |
| ---------- | -------------- | ---- | ------------ | -------------------------- |
| B001E4KFG0 | A3SGXH7AUHU8GW | 5    | 好品质狗粮。 | 我已经买了几罐活力狗粮……   |
| B00813GRG4 | A1D87F6ZCVE5NK | 1    | 不如广告。   | 产品标签写着巨型盐渍花生…… |

我们将评论概要和评论文本组合成单个组合文本。模型将对这个组合文本进行编码，输出单个向量 embedding。

[获取数据集](https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb)

```python
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df.to_csv('output/embedded_1k_reviews.csv', index=False)
```

如果想要从已保存的文件中加载数据，你可以这样：

```python
import pandas as pd

df = pd.read_csv('output/embedded_1k_reviews.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
```

:::details 数据 2D 可视化
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_2D.ipynb" target="_blank" rel="noreferrer" class="tag-link">Visualizing_embeddings_in_2D.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>  
Embedding 的大小取决于底层模型的复杂性。为了可视化这个高维数据，我们使用 t-SNE 算法将数据转换为二维。我们根据评价者给出的星级评价将各个评论着色：

1 星：红色  
2 星：深橙色  
3 星：金色  
4 星：青绿色  
5 星：深绿色

<picture><source type="image/webp" srcset="https://cdn.openai.com/API/docs/images/embeddings-tsne.webp"><source type="image/png" srcset="https://cdn.openai.com/API/docs/images/embeddings-tsne.png"><img src="https://cdn.openai.com/API/docs/images/embeddings-tsne.png" alt="Amazon ratings visualized in language using t-SNE" width="414" height="290"></picture>

可视化似乎产生了大约 3 个聚类簇，其中一个聚类簇主要由消极评价组成。

```python
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv('output/embedded_1k_reviews.csv')
matrix = df.ada_embedding.apply(eval).to_list()

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

colors = ["red", "darkorange", "gold", "turquiose", "darkgreen"]
x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]
color_indices = df.Score.values - 1

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
plt.title("Amazon ratings visualized in language using t-SNE")
```

:::

:::details Embedding 作为 ML 算法的文本特征编码器
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Regression_using_embeddings.ipynb" target="_blank" rel="noreferrer" class="tag-link">Regression_using_embeddings.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>

如果某些相关输入是自然文本，则引入 embedding 将提高任何机器学习模型的性能。embedding 还可以作为机器学习模型中的分类特征编码器。如果分类变量的名称具有意义且数量众多，例如职位名称，则价值最大。对于此任务，相似性 embedding 通常比搜索 embedding 表现更好。

我们观察到 embedding 表示通常非常丰富和信息密集。例如，使用 SVD 或 PCA 将输入的维数降低仅 10％，通常会导致特定任务的下游性能更差。

此代码将数据分成训练集和测试集，将分别用于下面的两个用例，即回归和分类。

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    list(df.ada_embedding.values),
    df.Score,
    test_size = 0.2,
    random_state=42
)
```

#### 使用 Embedding 特征的回归

Embedding 提供了一种优雅的预测数值值的方式。在此示例中，我们根据评论文本预测评论者的星级评分。由于 Embedding 中包含的语义信息很高，即使只有很少的评论，预测也是不错的。

我们假设分数是 1 到 5 之间的连续变量，并允许算法预测任何浮点值。机器学习算法将预测值与真实分数的距离最小化，并取得了均方误差为 0.39 的效果，这意味着平均预测偏差不到半星。

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train，y_train)
preds = rfr.predict(X_test)
```

:::

:::details 利用 embedding 特性进行文本分类
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb" target="_blank" rel="noreferrer" class="tag-link">Classification_using_embeddings.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>

这次，我们将尝试将算法分类为 5 个桶，从 1 星到 5 星，而不是预测 1 到 5 之间的任何值来预测评论中的确切星级数量。

在训练后，该模型学会了更好地预测 1 星和 5 星的评论，而不是更细微差别（2-4 星）的评论，可能由于更极端的情感表达造成。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
```

:::

:::details zero-shot 分类
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb" target="_blank" rel="noreferrer" class="tag-link">Zero-shot_classification_with_embeddings.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>  
我们可以使用 embedding 进行零样本分类，无需任何标注训练数据。对于每个类别，我们将类名或类别简短描述进行嵌入。在零样本分类中，我们将待分类的新文本与所有类别的嵌入进行比较，预测相似度最高的类别。

```python
from openai.embeddings_utils import cosine_similarity, get_embedding

df= df[df.Score!=3]
df['sentiment'] = df.Score.replace({1:'negative', 2:'negative', 4:'positive', 5:'positive'})

labels = ['negative', 'positive']
label_embeddings = [get_embedding(label, model=model) for label in labels]

def label_score(review_embedding, label_embeddings):
   return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

prediction = 'positive' if label_score('Sample Review', label_embeddings) > 0 else 'negative'
```

:::

:::details 为冷启动建议获得用户和产品 embedding
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/User_and_product_embeddings.ipynb" target="_blank" rel="noreferrer" class="tag-link">User_and_product_embeddings.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>  
我们可以通过对用户所有评论求平均值来获得用户嵌入。类似地，我们可以通过对围绕该产品的所有评论进行平均值来获得产品嵌入。为展示这种方法的有用性，我们使用了一个包含 50k 条评论的子集，以涵盖更多用户和产品的评论。

我们在一个单独的测试集上评估这些嵌入的有用性，我们以评分作为函数来绘制用户和产品嵌入的相似性。有趣的是，在这种方法的基础上，即使在用户收到产品之前，我们也可以比随机更好地预测他们是否会喜欢该产品。
<picture><source type="image/webp" srcset="https://cdn.openai.com/API/docs/images/embeddings-boxplot.webp"><source type="image/png" srcset="https://cdn.openai.com/API/docs/images/embeddings-boxplot.png"><img src="https://cdn.openai.com/API/docs/images/embeddings-boxplot.png" alt="Boxplot grouped by Score" width="420" height="312"></picture>

```python
user_embeddings = df.groupby('UserId').ada_embedding.apply(np.mean)
prod_embeddings = df.groupby('ProductId').ada_embedding.apply(np.mean)
```

:::

:::details 聚类
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb" target="_blank" rel="noreferrer" class="tag-link">Clustering.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>  
聚类是理解大量文本数据的一种方法。嵌入对于此任务非常有用，因为它们提供了每个文本的语义有意义的向量表示。因此，在无监督的方式中，聚类将揭示数据集中隐藏的分组。

在这个例子中，我们发现了四个明显的聚类：一个聚焦于狗粮，一个聚焦于负面评论，以及两个聚焦于正面评论。

<picture><source type="image/webp" srcset="https://cdn.openai.com/API/docs/images/embeddings-cluster.webp"><source type="image/png" srcset="https://cdn.openai.com/API/docs/images/embeddings-cluster.png"><img src="https://cdn.openai.com/API/docs/images/embeddings-cluster.png" alt="Clusters identified visualized in language 2d using t-SNE" width="418" height="290"></picture>

```python
import numpy as np
from sklearn.cluster import KMeans

matrix = np.vstack(df.ada_embedding.values)
n_clusters = 4

kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
df['Cluster'] = kmeans.labels_
```

:::

:::details 用 embeddings 进行文本搜索
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb" target="_blank" rel="noreferrer" class="tag-link">Semantic_text_search_using_embeddings.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>

为了检索最相关的文档，我们使用查询嵌入向量和每个文档之间的余弦相似度，并返回得分最高的文档。

```python
from openai.embeddings_utils import get_embedding, cosine_similarity

def search_reviews(df, product_description, n=3, pprint=True):
 embedding = get_embedding(product_description, model='text-embedding-ada-002')
 df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
 res = df.sort_values('similarities', ascending=False).head(n)
 return res

res = search_reviews(df, 'delicious beans', n=3)
```

:::

:::details 用 embeddings 代码搜索
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Code_search.ipynb" target="_blank" rel="noreferrer" class="tag-link">Code_search.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>  
使用嵌入进行代码与基于嵌入的文本搜索类似。我们提供了一种方法，从给定资料库中的所有 Python 文件中提取 Python 函数。然后，每个函数都由文本嵌入 ada-002 模型索引。

为了进行代码搜索，我们使用相同的模型将查询以自然语言嵌入式编码。然后，我们计算结果查询嵌入和每个函数嵌入之间的余弦相似性。最高的余弦相似度结果最相关。

```python
from openai.embeddings_utils import get_embedding, cosine_similarity

df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
   embedding = get_embedding(code_query, model='text-embedding-ada-002')
   df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

   res = df.sort_values('similarities', ascending=False).head(n)
   return res
res = search_functions(df, 'Completions API tests', n=3)
```

:::

:::details 用 embeddings 进行推荐
<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb" target="_blank" rel="noreferrer" class="tag-link">Recommendation_using_embeddings.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>  
因为嵌入向量之间的距离越短，表示它们越相似，所以嵌入可以用于推荐系统。下面，我们展示一个基本的推荐系统。它输入一个字符串列表和一个“源”字符串，计算它们的嵌入，然后返回一个字符串排名，按照与源字符串最相似到最不相似的顺序进行排列。  
作为一个具体的例子，下面的链接笔记本将此功能的一个版本应用于 [AG 新闻数据集](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)（缩减到 2000 个新闻文章描述），以返回与任何给定源文章最相似的前 5 篇文章。

```python
def recommendations_from_strings(
   strings: List[str],
   index_of_source_string: int,
   model="text-embedding-ada-002",
) -> List[int]:
   """Return nearest neighbors of a given string."""

   # get embeddings for all strings
   embeddings = [embedding_from_string(string, model=model) for string in strings]

   # get the embedding of the source string
   query_embedding = embeddings[index_of_source_string]

   # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
   distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

   # get indices of nearest neighbors (function from embeddings_utils.py)
   indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
   return indices_of_nearest_neighbors
```

:::

## 限制和风险

我们的 embedding 模型在某些情况下可能不可靠或存在社会风险，并且在没有补救措施的情况下可能会造成伤害。

### 社会偏见

:::info 限制
The models encode social biases, e.g. via stereotypes or negative sentiment towards certain groups.
模型可能编码社会偏见，例如通过刻板印象或对某些群体的负面情绪。
:::

我们通过运行 SEAT（[May et al，2019](https://arxiv.org/abs/1903.10561)）和 Winogender（[Rudinger et al，2018](https://arxiv.org/abs/1804.09301)）基准测试发现我们的模型存在偏见。这些基准测试包括 7 个测试，测试模型在应用于性别名称、地区名称和某些刻板印象时是否包含隐含偏见。

例如，我们发现我们的模型更强烈地将（a）欧裔美国人的姓名与积极情绪相关联，而相较于非裔美国人的姓名，（b）将负面刻板印象与黑人女性相关联。

这些基准测试在几个方面都有限制：（a）它们可能不适用于您的特定用例，以及（b）它们仅针对可能存在的极小一部分社会偏见进行测试。

这些测试是初步的，我们建议为您的特定用例运行测试。这些结果应被视为存在该现象的证据，而不是对您的用例的明确描述。有关更多详细信息和指导，请参见我们的[使用政策](https://openai.com/policies/usage-policies)。

如果您有任何问题，请[通过聊天联系我们的支持团队](https://help.openai.com/en/)；我们很乐意为您提供建议。

对最近事件的缺失
:::info 限制

模型缺乏对 2020 年 8 月之后发生的事件的了解。
:::
我们的模型是基于数据集训练的，这些数据集包含有关实际世界事件的某些信息，直到 8/2020。如果您依赖于模型代表最近事件，那么它们可能表现不佳。

## 常见问题

### 我如何在 embedding 字符串之前查看它有多少个令牌？

在 Python 中，您可以使用 OpenAI 的分词器 tiktoken 将字符串拆分为令牌。

示例代码：

```python
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
"""返回文本字符串中令牌的数量。"""
encoding = tiktoken.get_encoding(encoding_name)
num_tokens = len(encoding.encode(string))
return num_tokens

num_tokens_from_string("tiktoken is great!", "cl100k_base")
```

对于像 `text-embedding-ada-002` 这样的第二代嵌入模型，请使用 `cl100k_base` 编码。

有关更多详细信息和示例代码，请参见 OpenAI Cookbook 指南[如何使用 tiktoken 计数令牌](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)。

### 我如何快速检索 K 个最近的嵌入向量？

为了快速搜索多个向量，我们推荐使用向量数据库。在我们的 GitHub 上的 [Cookbook](https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases) 中，您可以找到使用向量数据库和 OpenAI API 的工作示例。

向量数据库选项包括：

- Pinecone，一个完全管理的向量数据库
- Weaviate，一个开源的向量搜索引擎
- Redis 作为向量数据库
- Qdrant，一个向量搜索引擎
- Milvus，用于可扩展相似度搜索的向量数据库
- Chroma，一个开源的嵌入存储
- Typesense，快速的开源向量搜索
- Zilliz，由 Milvus 提供技术支持的数据基础设施

### 我应该使用哪个距离函数？

我们建议使用[余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)。距离函数的选择通常并不重要。

OpenAI embeddings 被统一为长度 1，这意味着：

- 可以使用点积更快地计算余弦相似度
- 余弦相似度和欧几里得距离将产生相同的排名

### 我可以在线共享我的嵌入向量吗？

客户拥有我们的模型的输入和输出，包括 embeddings。您有责任确保您输入到我们的 API 中的内容不违反任何适用法律或我们的[使用条款](https://openai.com/policies/terms-of-use)。
