# 如何构建一个能回答有关您的网站的问题的人工智能

本教程演示了如何爬取网站（本例中为 OpenAI 网站），使用 [Embeddings API](https://platform.openai.com/docs/guides/embeddings) 将爬取的页面转换为向量，并创建基本的搜索功能，以允许用户提出有关向量中信息的问题。这旨在为更复杂的应用程序提供一个引导，这些应用程序可以使用自定义的知识库。

## 入门准备

本教程需要一些 Python 和 GitHub 的基础知识。在开始之前，请确保[设置了 OpenAI API 密钥](https://platform.openai.com/docs/api-reference/introduction)并完成了[快速入门教程](https://platform.openai.com/docs/quickstart)。这将让你有一个关于如何充分利用 API 的良好感觉。

Python 作为主要编程语言，与 OpenAI、Pandas、transformers、NumPy 和其他流行的包一起使用。如果在处理本教程时遇到任何问题，请在 [OpenAI 社区论坛](https://community.openai.com/)上提问。

要开始使用代码，请在 GitHub 上[克隆本教程的全部代码](https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a)。或者，跟随并将每个部分复制到 Jupyter 笔记本中，并按步骤运行代码，或者只是阅读。避免任何问题的好方法是设置一个新的虚拟环境，并通过运行以下命令安装所需的软件包：

```bash
python -m venv env

source env/bin/activate

pip install -r requirements.txt
```

## 建立网络爬虫

本教程的主要重点是 OpenAI API，因此如果您愿意，可以跳过有关如何创建网络爬虫的内容并只[下载源代码](https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a)。否则，请扩展以下部分以了解爬取机制的实现。

:::details 学习如何构建网络爬虫
Acquiring data in text form is the first step to use embeddings. This tutorial creates a new set of data by crawling the OpenAI website, a technique that you can also use for your own company or personal website.

获取文本形式的数据是使用嵌入的第一步。本教程通过爬取 OpenAI 网站创建了一个新的数据集，这是一种您也可以用于您自己公司或个人网站的技术。
[查看源代码](https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a)

尽管这个爬虫是完全从头开始写的，但是像 Scrapy 这样的开源包也可以帮助进行这些操作。

此爬虫将从下面代码底部传入的根 URL 开始，访问每个页面，查找其他链接，并访问那些页面（只要它们具有相同的根域）。开始前，请导入所需的软件包，设置基本 URL，并定义 HTMLParser 类。

```python
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

domain = "openai.com" # <- put your domain to be crawled
full_url = "https://openai.com/" # <- put your domain to be crawled with https or http

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])
```

下一个函数接受一个 URL 作为参数，打开该 URL 并读取 HTML 内容。然后，它返回在该页面上发现的所有超链接。

```python
# Function to get the hyperlinks from a URL
def get_hyperlinks(url):

    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks
```

目标是仅爬取和索引 OpenAI 域下的内容。为此，需要一个调用 get_hyperlinks 函数但过滤掉任何不属于指定域的 URL 的函数。

```python
# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))
```

`crawl`函数是网页抓取任务的最后一步。它会跟踪访问的 URL，以避免重复访问相同的页面，这可能会链接到网站上的多个页面。它还可以提取页面中没有 HTML 标签的原始文本，并将文本内容写入特定于页面的本地 .txt 文件中。

```python
    def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

crawl(full_url)
```

以上示例的最后一行运行爬虫程序，该程序会遍历所有可访问的链接并将这些页面转换成文本文件。运行时间会根据您的网站规模和复杂程度而有所不同，可能需要几分钟时间。
:::

## 构建向量索引

CSV 是存储嵌入的常见格式。您可以通过将原始文本文件（位于 text 目录中）转换为 Pandas 数据框来使用 Python 中的格式。Pandas 是一种流行的开源库，可帮助您处理表格数据（以行和列存储的数据）。
空白空行可能会混淆文本文件并使它们更难处理。一个简单的函数可以去除这些行并整理文件。

```python
def remove_newlines(serie):
serie = serie.str.replace('\n', ' ')
serie = serie.str.replace('\\n', ' ')
serie = serie.str.replace(' ', ' ')
serie = serie.str.replace(' ', ' ')
return serie
```

将文本转换为 CSV 需要循环遍历先前创建的文本目录中的文本文件。打开每个文件后，删除额外的空格并将修改后的文本附加到列表中。然后，将去除新行的文本添加到空的 Pandas 数据帧中，然后将数据帧写入 CSV 文件。

:::tip 提示
额外的空格和新行可能会使文本杂乱无章并复杂化嵌入过程。这里使用的代码有助于去除其中一些空格，但您可能会发现第三方库或其他方法有用，可以获得更多不必要的字符。
:::

```python
import pandas as pd

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()
```

标记化是将原始文本保存到 CSV 文件后的下一步。这个过程通过分解句子和单词将输入文本拆分为标记。您可以通过[查看我们文档中的 Tokenizer ](https://platform.openai.com/tokenizer)来进行可视化演示。

:::tip 提示
一个有用的经验法则是，对于普通的英语文本，一个标记通常对应于 ~4 个字符的文本。这相当于大约 ¾ 个单词（因此 100 个标记~= 75 个单词）。
:::

API 对于嵌入的最大输入标记数有限制。为了保持在限制以下，CSV 文件中的文本需要分解成多行。首先记录每行的现有长度，以确定哪些行需要分解。

```python
import tiktoken

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()
```

<div class="sandbox-preview"><div class="sandbox-screenshot"><img src="https://cdn.openai.com/API/docs/images/tutorials/web-qa/embeddings-initial-histrogram.png" alt="Embeddings histogram" width="553" height="413"></div></div>
将文本分解为 CSV 后，最新的嵌入模型可以处理具有多达 8191 个输入标记的输入，因此大多数行不需要分解，但在抓取的每个子页面上可能不是这种情况，因此下一个代码段将长行拆分为更小的块。

```python
max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens

def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

shortened = []

# Loop through the dataframe

for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )

```

重新可视化的直方图可以帮助确认行是否成功分成了更短的部分。

```python
df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()
```

<div class="sandbox-preview"><div class="sandbox-screenshot"><img src="https://cdn.openai.com/API/docs/images/tutorials/web-qa/embeddings-tokenized-output.png" alt="Embeddings tokenized output" width="552" height="418"></div></div>

现在，文本已经被分解为更小的块，可以向 OpenAI API 发送一个简单的请求，指定使用新的 text-embedding-ada-002 模型创建向量：

```python
import openai

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

df.to_csv('processed/embeddings.csv')
df.head()
```

这应该需要大约 3 至 5 分钟，之后您将拥有可以使用的向量！

## 使用你的向量构建一个问答系统

准备好向量之后，这个过程的最后一步是创建一个简单的问答系统。这将接受用户的问题，创建其向量，并将其与现有向量进行比较，以从抓取的网站中检索最相关的文本。然后，`text-davinci-003` 模型将根据检索到的文本生成自然语言回答。

将向量转换为 NumPy 数组是第一步，因为它会提供更多灵活性，使其更适合使用许多操作 NumPy 数组的可用函数。它还会将维度扁平化为 1-D，这是许多后续操作所需的格式。

```python
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()
```

问题需要使用一个简单的函数转换为向量，现在数据已经准备好了。这很重要，因为使用余弦距离比较向量的数字（这是原始文本的变形）的搜索可能会相似并且可能是问题的答案，如果它们的余弦距离接近。 OpenAI python 包在这里有一个内置的 distances_from_embeddings 函数，非常有用。

```python
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)
```

将文本分成较小的单词集，以升序循环并继续添加文本，这是确保答案完整的关键步骤。如果返回的内容超出限制，max_len 还可以修改为较小的值。

前一步只检索与问题有语义关联的文本块，因此它们可能包含答案，但不保证。通过返回前 5 个最有可能的结果，可以进一步增加发现答案的机会。

回答提示将尝试从检索到的上下文中提取相关事实，以形成连贯的答案。如果没有相关答案，提示将返回“我不知道”。

可以使用`text-davinci-003`的完成端点创建一个听起来真实的答案。

```python
def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
```

完成了！一个具有从 OpenAI 网站知识的问答系统现已准备就绪。可以进行一些快速测试以查看输出的质量：

```python
answer_question(df, question="今天是星期几？", debug=False)

answer_question(df, question="我们最新的嵌入模型是什么？")

answer_question(df, question="什么是ChatGPT？")
```

响应将类似于以下内容：

```python
"我不知道。"

'最新的嵌入模型是文本嵌入 ada-002。'

'ChatGPT 是一个训练用于以会话方式交互的模型。它能回答后续问题，承认错误，挑战不正确的前提条件并拒绝不适当的请求。'
```

如果系统无法回答预期的问题，最好检索一下原始文本文件，以查看预期的信息是否实际上被转化了。最初进行的抓取过程被设置为跳过所提供的原始域外的站点，因此，如果有子域设置，则可能没有该知识。

目前，每次回答问题时都将数据框架传递进去。对于生产工作流程，应使用[向量数据库](https://platform.openai.com/docs/guides/embeddings/how-can-i-retrieve-k-nearest-embedding-vectors-quickly)解决方案，而不是将嵌入存储在 CSV 文件中，但当前的方法用来检验一个原型的好选择。
