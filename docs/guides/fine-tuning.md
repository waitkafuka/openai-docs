# 模型微调（Fine-tuning）

学习如何为你的应用定制一个模型。

## 介绍

让你能够定制化 API 中提供的模型，从而获得以下好处：

1. 比使用 prompt 获得更高质量的结果。
2. 相对于 prompt 来说，能够在更多的示例上进行训练
3. 可以使用较短的 prompt，从而节省 token
4. 更低的延迟请求

GPT-3 已经在大量来自开放互联网的文本上进行了预训练。当只提供少量示例的提示时，它通常可以直观地确定您要执行的任务并生成一个可信的补全。这通常称为“少量样本学习”。

相对于”少量样本学习“，Fine-tuning 在许多示例上进行训练，比快速训练可以获得更好的结果，适用于多种任务。一旦 Fine-tuning 微调了模型，您就不需要再在提示中提供示例了。这可以节省成本，并实现较低延迟请求。

概括来说，Fine-tuning 的动作包括以下步骤：

1. 准备和上传训练数据。
2. 训练新的 Fine-tuning 模型。
3. 使用您的 Fine-tuning 模型。

访问我们的[价格页面](https://openai.com/api/pricing)了解更多关于模型训练和使用的费用信息。

## 哪些模型可以进行 Fine-tuning？

Fine-tuning 目前仅适用于以下基础模型：`davinci`、`curie`、`babbage` 和 `ada`。这些是没有任何训练后指令的原始模型（如 `text-davinci-003`）。您也可以继续 Fine-tuning 一个已[经过 Fine-tuning 的模型](https://platform.openai.com/docs/guides/fine-tuning/continue-fine-tuning-from-a-fine-tuned-model)，以添加其他数据，而无需从头开始。

## 安装

我们建议使用 OpenAI 命令行界面（CLI）。安装方法：运行：

```bash
pip install -- upgrade openai
```

（下面的介绍适用于 0.9.4 版本以上。另外，OpenAI CLI 需要 python3）

使用之前，需设置 OPENAI_API_KEY 环境变量。

```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

## 准备训练数据

训练数据是教 GPT-3 您想教给 GPT-3 学习的数据。

您的数据必须是一个 [JSONL](https://jsonlines.org/) 文档，其中每行都是与一个训练示例相对应的提示-完成对。您可以使用我们的 [CLI 数据准备工具](https://platform.openai.com/docs/guides/fine-tuning/cli-data-preparation-tool)轻松将数据转换为此文件格式。

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

训练用的 prompt 和调用模型接口（如 Davinci、Curie、Babbage、Ada）使用的 prompt 设计方法不尽相同。特别是，基本模型的提示通常由多个示例组成（"少量学习:few-shot learning"），而对于微调，每个训练示例通常由单个输入示例及其关联的输出组成，无需给出详细说明或在同一提示中包含多个示例。

有关如何为各种任务准备训练数据的更详细指导，请参考我们的[准备数据集](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)的最佳实践。

训练数据集多多益善。我们建议您至少有几百个示例。一般来说，我们发现数据集大小每翻倍，模型质量将会线性增加。

CLI 数据集准备工具

我们开发了一个工具，可以验证、给出建议并重新格式化您的数据，以准备进行 Fine-tuning。

```bash
openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
```

此工具可接受不同格式的文件，唯一的要求是它们需要包含提示和完成列/键。您可以传递 **CSV、TSV、XLSX、JSON** 或 **JSONL** 文件，在建议你修改其中的问题之后，它将把输出保存为 JSONL 文件，以备进行微调。

## 创建一个 fine-tuned 模型

以下假设你已经按照上述说明准备好了训练数据。

使用 OpenAI CLI 启动你的微调作业：

```bash

openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
```

其中，`BASE_MODEL` 是你要从中开始的基础模型的名称（ada、 babbage、 curie 或 davinci）。你可以使用[后缀参数](https://platform.openai.com/docs/guides/fine-tuning/customize-your-model-name)自定义你微调后的模型名称。

运行上述命令会执行以下操作：

1. 使用[文件 API](https://platform.openai.com/docs/api-reference/files) 上传文件（或使用已上传的文件）
2. 创建微调作业
3. 流式传输事件，直到作业完成（这通常需要几分钟，但如果队列中有许多作业或您的数据集很大，则可能需要几个小时）

每个微调作业都从一个基础模型开始，默认为 curie。模型的选择会影响模型的性能和运行微调后的模型的成本。你的模型可以是：`ada`、`babbage`、`curie` 或 `davinci`。访问我们的[定价页面](https://openai.com/api/pricing/#faq-fine-tuning-pricing-calculation)了解微调费率的详细信息。

启动微调作业后，可能需要一些时间才能完成。你的作业可能在我们系统中排队等待，训练模型的时间取决于模型和数据集的大小，可能需要几分钟或几个小时。如果任何原因导致事件流被中断，你可以通过运行以下命令恢复：

```bash
openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
```

当任务完成后，它会显示微调后的模型名称。

除了创建微调作业之外，您还可以列出现有的作业，检索作业的状态或取消作业。

```bash
    # List all created fine-tunes
openai api fine_tunes.list

# Retrieve the state of a fine-tune. The resulting object includes
# job status (which can be one of pending, running, succeeded, or failed)
# and other information
openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# Cancel a job
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>
```

## 使用一个微调后的模型

当作业成功后，`fine_tuned_model` 字段将会显示模型名称。现在，您可以将此模型指定为我们的 Completions API 的一个参数，并使用 Playground 对其进行请求。

在您的工作完成后，您的模型可能需要几分钟才能准备好处理请求。如果向您的模型发出完整请求超时，则可能是因为您的模型仍在加载中。如果发生这种情况，请稍后再试。

您可以通过将模型名称作为完整请求的模型参数来开始发出请求：

OpenAI CLI:

```bash
openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>
```

cURL:

```bash
curl https://api.openai.com/v1/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": YOUR_PROMPT, "model": FINE_TUNED_MODEL}'
```

Python:

```python
import openai

openai.Completion.create(
model=FINE_TUNED_MODEL,
prompt=YOUR_PROMPT)
```

Node.js:

```javascript
const response = await openai.createCompletion({

model: FINE_TUNED_MODEL
prompt: YOUR_PROMPT,
});

```

在后面，在接口的请求中，你仍然可以使用其他[API 参数](https://platform.openai.com/docs/api-reference/completions)，例如：`temperature`, `frequency_penalty`, `presence_penalty`等等。

## 删除一个微调过的模型

为了删除一个模型，你必须在你的组织中是一个`owner`角色。

OpenAI CLI:

```bash
openai api models.delete -i <FINE_TUNED_MODEL>
```

cURL:

```bash
curl -X "DELETE" https://api.openai.com/v1/models/<FINE_TUNED_MODEL> \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

Python:

```python
import openai
openai.Model.delete(FINE_TUNED_MODEL)
```

# 准备你的数据集

微调是创建特定于您用例的新模型的强大技术。在**微调模型之前，我们强烈建议您阅读以下最佳实践和[特定用例](https://platform.openai.com/docs/guides/fine-tuning/specific-guidelines)的指南**。

## 数据格式

为微调模型，您需要一组训练示例，每个示例都包含一个单独的输入（“提示”）和其关联的输出（“完成”）。这与使用我们的基本模型有很大不同：在那里您可能会在单个提示中输入详细的说明或多个示例。

- 每个提示应以固定的分隔符结尾，以通知模型提示何时结束和完成何时开始。一个通常效果良好的简单分隔符是\n\n###\n\n。分隔符不应出现在任何提示中的其他位置。

- 每个完成应以空格开头，由于我们的标记化，它将大多数单词与前一个空格标记化。

- 每个完成应以固定的停止序列结尾，以通知模型何时结束。停止序列可以是\n、###或任何不出现在任何完成中的其他标记。

- 对于推理，您应以与创建训练数据集时相同的方式格式化提示，包括相同的分隔符。还要指定相同的停止序列以正确截断完成。

## 一般最佳实践

微调使用更多高质量示例可以获得更好的性能。为了微调比使用高质量提示与我们的基本模型性能更好的模型，您应该提供至少几百个高质量示例，最好由人类专家审查。从那里开始，性能往往会随着示例数量的加倍而线性增加。增加示例数量通常是提高性能的最佳且最可靠的方法。

分类器是最容易入手的模型。对于分类问题，我们建议使用 `ada`，它的表现通常只比专业模型稍差，同时速度和价格更快更便宜。

如果您正在微调现有数据集而不是从头编写提示，请确保尽可能手动审查数据，以查看是否存在冒犯性或不准确的内容，如果可能的话，或者如果数据集很大，尽可能审查尽可能多的随机样本。

## 具体指南

微调可以解决各种问题，最佳使用方式可能取决于您的具体用例。下面，我们列出微调的最常见用途和相应的指南。

<div class="ft-guide-toc"><ul><li><a href="/docs/guides/fine-tuning/classification">分类</a><ul><li><a href="/docs/guides/fine-tuning/case-study-is-the-model-making-untrue-statements">模型是否发表不实之词？</a></li><li><a href="/docs/guides/fine-tuning/case-study-sentiment-analysis">情感分析</a></li><li><a href="/docs/guides/fine-tuning/case-study-categorization-for-email-triage">用于电子邮件筛选的分类</a></li></ul></li><li><a href="/docs/guides/fine-tuning/conditional-generation">基于条件的生成</a><ul><li><a href="/docs/guides/fine-tuning/case-study-write-an-engaging-ad-based-on-a-wikipedia-article">基于维基百科文章编写引人入胜的广告</a></li><li><a href="/docs/guides/fine-tuning/case-study-entity-extraction">实体提取</a></li><li><a href="/docs/guides/fine-tuning/case-study-customer-support-chatbot">客户支持聊天机器人</a></li><li><a href="/docs/guides/fine-tuning/case-study-product-description-based-on-a-technical-list-of-properties">基于属性技术列表的产品描述</a></li></ul></li></ul></div>

## 分类

在分类问题中，提示中的每个输入应分类到预定义的类别中。对于这种类型的问题，我们建议：

- 在提示末尾使用分隔符，例如`\n\n###\n\n`。记得在您最终向模型发出请求时也附加此分隔符。
- 选择映射到单个标记的类别。在推理时，指定 max_tokens=1，因为您只需要分类的第一个标记。
- 确保提示+完成不超过 2048 个标记，包括分隔符
- 每个类别至少要有约 100 个示例
- 在使用模型时，您可以指定 logprobs=5（对于 5 个类别）以获取类别日志概率
- 确保用于微调的数据集在结构和任务类型上与模型将要用于的非常相似

## 学习案例：模型是否发表不实之词？

假设您想确保网站上的广告文本提到正确的产品和公司。换句话说，您希望确保模型不会胡编乱造。您可能希望微调一个分类器，过滤掉不正确的广告。

数据集可能如下所示：

```json
{"prompt":"Company: BHFF insurance\nProduct: allround insurance\nAd:One stop shop for all your insurance needs!\nSupported:", "completion":" yes"}
{"prompt":"Company: Loft conversion specialists\nProduct: -\nAd:Straight teeth in weeks!\nSupported:", "completion":" no"}
```

在上面的例子中，我们使用了一个结构化的输入，包含了公司名称、产品和相关广告。我们使用换行符`\nSupported:`作为分隔符，清晰地将提示与完成部分分开。如果有足够数量的示例，分隔符不会产生太大影响（通常不超过 0.4%），只要它不出现在提示或完成部分内部。

针对这种用例，我们调整了一个 `ada` 模型，因为它会更快、更便宜，而且性能与更大的模型相比是可以比拟的，因为这是一个分类任务。

现在，我们可以通过进行完成请求来查询我们的模型。

```bash
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "Company: Reliable accountants Ltd\nProduct: Personal Tax help\nAd:Best advice in town!\nSupported:",
    "max_tokens": 1,
    "model": "YOUR_FINE_TUNED_MODEL_NAME"
  }'
```

上面将会返回`yes`或者`no`。

## 学习案例：语句情感分析

在这个案例中，假设您想要知道特定推文的积极或消极程度。数据集可能如下所示：

```json
{"prompt":"Overjoyed with the new iPhone! ->", "completion":" positive"}
{"prompt":"@lakers disappoint for a third straight night https://t.co/38EFe43 ->", "completion":" negative"}
```

一旦模型调整好了，您就可以通过在完成请求上设置 `logprobs=2` 来获取第一个完成标记的 log 概率。正面类的概率越高，情感程度就越高。

现在，我们可以通过进行完成请求来查询我们的模型。

```bash
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "https://t.co/f93xEd2 Excited to share my latest blog post! ->",
    "max_tokens": 1,
    "model": "YOUR_FINE_TUNED_MODEL_NAME"
  }'
```

以上将会返回类似下面的结果：

```json
{
  "id": "cmpl-COMPLETION_ID",
  "object": "text_completion",
  "created": 1589498378,
  "model": "YOUR_FINE_TUNED_MODEL_NAME",
  "choices": [
    {
      "logprobs": {
        "text_offset": [19],
        "token_logprobs": [-0.03597255],
        "tokens": [" positive"],
        "top_logprobs": [
          {
            " negative": -4.9785037,
            " positive": -0.03597255
          }
        ]
      },

      "text": " positive",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

## 学习案例：邮件分类

假设您想将收到的电子邮件分类到众多预定义类别中的一种。对于分类到大量类别中，我们建议您将这些类别转换为数字，这在 ~500 个类别之内效果良好。我们已经观察到，在数字前面添加空格有时因为分词而略微提高了性能。您可能想按以下方式构建您的训练数据：

```json
{ "prompt": "Subject: <email_subject>\nFrom:<customer_name>\nDate:<date>\nContent:<email_body>\n\n###\n\n", "completion": " <numerical_category>" }
```

例如：

```json
{ "prompt": "Subject: Update my address\nFrom:Joe Doe\nTo:support@ourcompany.com\nDate:2021-06-03\nContent:Hi,\nI would like to update my billing address to match my delivery address.\n\nPlease let me know once done.\n\nThanks,\nJoe\n\n###\n\n", "completion": " 4" }
```

在上面的示例中，我们使用了一个最大为 2043 个 tokens 的邮箱作为输入。（这允许使用 4 个 token 的分隔符和一个 token 的完成，总计 2048 个。）我们使用的分隔符是`\n\n###\n\n`，并且删除了邮件中所有`###`的出现。

## 条件生成

条件生成是一个需要根据某种输入生成内容的场景。这包括改述、摘要、实体提取、根据规格书编写产品描述、聊天机器人等等。针对这种类型的问题，我们建议：

- 在提示的末尾使用分隔符，例如`\n\n###\n\n`。当您最终向您的模型发出请求时，请确保还附加了此分隔符。
- 在完成时使用一个终止标记，例如 `END`
- 在推理期间将结束标记添加为停止序列，例如 `stop=[" END"]`
- 力求拥有至少~500 个样例
- 确保提示+完成不超过 2048 个 token，包括分隔符
- 确保样例的质量高且遵循相同的期望格式
- 确保用于微调的数据集在结构和任务类型上与模型将要使用的数据集非常相似
- 对于这些用例，使用较低的学习率和仅 1-2 个 epoch 通常效果更好

## 学习案例：根据维基百科文章编写一个吸引人的广告

这是一个生成性用例，因此您需要确保提供的样本具有最高的质量，因为经过微调的模型将尝试模仿给定示例的风格（和错误）。最好是大约 500 个例子。一个示例数据集可能如下所示：

```json
{ "prompt": "<Product Name>\n<Wikipedia description>\n\n###\n\n", "completion": " <engaging ad> END" }
```

例如：

```json
{
  "prompt": "Samsung Galaxy Feel\nThe Samsung Galaxy Feel is an Android smartphone developed by Samsung Electronics exclusively for the Japanese market. The phone was released in June 2017 and was sold by NTT Docomo. It runs on Android 7.0 (Nougat), has a 4.7 inch display, and a 3000 mAh battery.\nSoftware\nSamsung Galaxy Feel runs on Android 7.0 (Nougat), but can be later updated to Android 8.0 (Oreo).\nHardware\nSamsung Galaxy Feel has a 4.7 inch Super AMOLED HD display, 16 MP back facing and 5 MP front facing cameras. It has a 3000 mAh battery, a 1.6 GHz Octa-Core ARM Cortex-A53 CPU, and an ARM Mali-T830 MP1 700 MHz GPU. It comes with 32GB of internal storage, expandable to 256GB via microSD. Aside from its software and hardware specifications, Samsung also introduced a unique a hole in the phone's shell to accommodate the Japanese perceived penchant for personalizing their mobile phones. The Galaxy Feel's battery was also touted as a major selling point since the market favors handsets with longer battery life. The device is also waterproof and supports 1seg digital broadcasts using an antenna that is sold separately.\n\n###\n\n",
  "completion": "Looking for a smartphone that can do it all? Look no further than Samsung Galaxy Feel! With a slim and sleek design, our latest smartphone features high-quality picture and video capabilities, as well as an award winning battery life. END"
}
```

在这里，我们使用了多行分隔符，因为维基百科文章包含多个段落和标题。我们还使用了一个简单的结束符号，以确保模型知道何时完成生成。

## 学习案例：实体提取

这类似于语言转换任务。为了提高性能，最好按字母顺序或按原始文本中出现的顺序分类不同提取的实体。这将有助于模型跟踪需要按顺序生成的所有实体。数据集可能如下所示：

```json
{ "prompt": "<any text, for example news article>\n\n###\n\n", "completion": " <list of entities, separated by a newline> END" }
```

例如：

```json
{ "prompt": "Portugal will be removed from the UK's green travel list from Tuesday, amid rising coronavirus cases and concern over a \"Nepal mutation of the so-called Indian variant\". It will join the amber list, meaning holidaymakers should not visit and returnees must isolate for 10 days...\n\n###\n\n", "completion": " Portugal\nUK\nNepal mutation\nIndian variant END" }
```

多行分隔符效果最佳，因为文本可能包含多行。理想情况下，输入提示的类型应具有高多样性，反映出在提取实体时可能会遇到的文本类型（新闻文章、维基百科页面、推文、法律文件）。

## 学习案例：客户支持聊天机器人

聊天机器人通常包含与对话相关的上下文（订单详细信息）、迄今为止对话的摘要以及最近的消息。对于这个用例，相同的过去对话可以在数据集中生成多行，每次生成时上下文都略有不同，每个代理都生成一个完成。这个用例将需要几千个例子，因为它可能涉及不同类型的请求和客户问题。为了确保性能高质量，我们建议审核会话样本以确保消息的质量。摘要可以用单独的文本转换微调模型生成。数据集可能如下所示：

```json
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent:", "completion":" <response2>\n"}
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent: <response2>\nCustomer: <message3>\nAgent:", "completion":" <response3>\n"}
```

在这里，我们有意分离了不同类型的输入信息，但在一个提示和完成之间保持了客户代理对话的相同格式。所有的完成都应该由代理完成，当进行推断时，我们可以使用`\n` 作为停止序列。

## 学习案例：基于技术属性清单的产品描述

在这里，将输入数据转换为自然语言是很重要的，这可能会导致更高的性能。例如，以下格式：

```json
{ "prompt": "Item=handbag, Color=army_green, price=$99, size=S->", "completion": " This stylish small green handbag will add a unique touch to your look, without costing you a fortune." }
```

下面这种是不行的：

```json
{ "prompt": "Item is a handbag. Colour is army green. Price is midrange. Size is small.->", "completion": " This stylish small green handbag will add a unique touch to your look, without costing you a fortune." }
```

为了高性能，确保完成是基于提供的描述完成的。如果经常查询外部内容，则以自动化的方式添加此类内容会提高性能。如果描述基于图像，则使用算法提取图像的文本描述可能有所帮助。由于完成只有一句话，因此我们在推断时可以使用`.`作为停止序列。

# 高级用法

## 自定义模型名称

您可以使用[后缀参数](https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-suffix)将长达 40 个字符的后缀添加到微调模型名称中。

OpenAI CLI:

```bash
openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name"
```

结果：

```
ada:ft-your-org:custom-model-name-2022-02-15-04-21-04
```

## 分析你的微调模型

我们将结果文件附加到每个作业中。当您检索微调时，将列出此结果文件的 ID，当您查看微调上的事件时也是如此。您可以下载这些文件：

OpenAI CLI：

```bash
openai api fine_tunes.results -i <YOUR_FINE_TUNE_JOB_ID>
```

CURL：

```bash
curl https://api.openai.com/v1/files/$RESULTS_FILE_ID/content \
 -H "Authorization: Bearer $OPENAI_API_KEY" > results.csv
```

`_results.csv`文件为每个训练步骤包含了一行，其中一步指对一批数据进行一次前向和后向传递。除了步骤编号，每行还包含与该步骤对应的以下字段：

- **elapsed_tokens**：模型到目前为止已经看到的记号数（包括重复）
- **elapsed_examples**：模型到目前为止已经看到的示例数（包括重复），其中一个示例是批处理中的一个元素。例如，如果 `batch_size = 4`，则每个步骤将使 `elapsed_examples` 增加 4。
- `training_loss`：训练批次的损失
- `training_sequence_accuracy`：训练批次中**完成率**的百分比，其中模型预测的 token 完全匹配真实完成 token。例如，如果 `batch_size` 为 3，如果您的数据包含完成[[1,2]，[0,5]，[4,2]]，并且模型预测[[1,1]，[0,5]，[4,2]]，则此准确度将为 2/3 = 0.67
- **training_token_accuracy**：模型正确预测的训练批次中的标记百分比。例如，如果 `batch_size` 为 3，如果您的数据包含完成[[1,2]，[0,5]，[4,2]]，并且模型预测[[1,1]，[0,5]，[4,2]]，则此准确度将为 5/6 = 0.83

## 分类特定指标

我们还提供生成结果文件中的其他分类特定指标选项，例如准确度和加权 F1 分数。这些指标定期针对完整的验证集计算，并在微调结束时计算。您将在结果文件中看到它们作为其他列。

要启用此选项，请设置参数`--compute_classification_metrics`。此外，您必须提供一个验证文件，如果是多类分类，设置`classification_n_classes` 参数；如果是二进制分类，设置`classification_positive_class`。

OpenAI CLI：

```bash
# For multiclass classification
openai api fine_tunes.create \
  -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_OR_PATH> \
  -m <MODEL> \
  --compute_classification_metrics \
  --classification_n_classes <N_CLASSES>

# For binary classification
openai api fine_tunes.create \
  -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_OR_PATH> \
  -m <MODEL> \
  --compute_classification_metrics \
  --classification_n_classes 2 \
  --classification_positive_class <POSITIVE_CLASS_FROM_DATASET>
```

如果你设置了 `--compute_classification_metrics`，以下指标将显示在你的[结果文件](https://platform.openai.com/docs/guides/fine-tuning/analyzing-your-fine-tuned-model)中：

**对于多类别分类**

- **classification/accuracy**：精确度
- **classification/weighted_f1_score**: 加权 F-1 分数

**对于二进制分类**

下列指标基于一个分类阀值为 0.5（即当概率 > 0.5 时，样本被归类为正类）。

**classification/accuracy**：准确率
**classification/precision**：精确率
**classification/recall**: 召回率
**classification/f{beta}**: 分数
**classification/auroc - AUROC**：曲线下面积
**classification/auprc - AUPRC**：准确率曲线下面积

需要注意的是，这些评估假设您使用的是文本标签来表示能够划分为单个记号的类别，如上面所述。如果不满足这些条件，则您得到的数字可能是错误的。

## 验证

你可以为验证保留一些数据。验证文件与训练文件具有完全相同的格式，你的训练和验证数据应该是互斥的。

如果在创建微调工作时提供了验证文件，则生成的结果文件将包括在训练期间定期评估模型在验证数据上表现的评估结果。

OpenAI CLI:

```bash
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> \
 -v <VALIDATION_FILE_ID_OR_PATH> \
 -m <MODEL>
```

如果提供了验证文件，我们会在训练时间对验证数据批次进行度量计算。您将在结果文件中看到以下额外的指标：

- **validation_loss**: 验证批次上的损失
- **validation_sequence_accuracy**: 模型预测的记号与真实完成记号完全匹配的完成在验证批次中的百分比。例如，如果您有一个批次大小为 3 的数据，其中包含完成[[1, 2]，[0，5]，[4，2]]，而模型预测[[1, 1]，[0，5]，[4，2]]，则准确率为 2/3 = 0.67。
- **validation_token_accuracy**: 模型正确预测验证批次中标记的百分比。例如，如果您有一个批次大小为 3 的数据，其中包含完成[[1, 2]，[0，5]，[4，2]]，而模型预测[[1, 1]，[0，5]，[4，2]]，则准确率为 5/6 = 0.83。

## 超级参数

我们已经选择了适用于各种使用情况的默认超参数。唯一需要的参数是训练文件。

话虽如此，调整用于微调的超参数通常会导致产生更高质量输出的模型。特别是，您可能想要配置以下内容：

- `model`：基本模型的名称以进行微调。您可以选择其中之一：“ada”，“babbage”，“curie”或“davinci”。要了解有关这些模型的更多信息，请参阅 [Models](https://platform.openai.com/docs/models) 文档。
- `n_epochs`-默认为 4。训练模型的纪元数量。一个时代指对训练数据集进行一次完整循环。
- `batch_size`-默认为训练集中示例数量的约 0.2％，上限为 256。批量大小是用于训练单个正向和反向传递的训练示例的数量。通常，我们发现较大的批量大小倾向于对较大的数据集效果更好。
- `learning_rate_multiplier`-默认为 0.05、0.1 或 0.2，具体取决于最终 batch_size。微调学习速率是用于预训练的原始学习速率乘以该乘数。我们建议尝试在 0.02 到 0.2 的范围内的值，以查看哪些值会产生最佳结果。根据经验，我们发现较大的学习速率在批量大小较大时通常表现更好。
- `compute_classification_metrics`-默认为 False。如果为 True，则用于分类任务的微调，在每个时代结束时计算分类特定指标（准确度，F-1 分数等）的验证集。
  要配置这些附加超参数，请通过 OpenAI CLI 上的命令行标志传递它们，例如：

```bash
openai api fine_tunes.create \
  -t file-JD89ePi5KMsB3Tayeli5ovfW \
  -m ada \
  --n_epochs 1
```

## 继续从微调模型微调

如果您已经对任务进行了微调，并且现在有其他训练数据需要纳入考虑，您可以继续从模型进行微调。这样创建的模型已经从所有训练数据中学习，无需从头开始重新训练。

为此，创建新的微调工作时，传递已经微调的模型名称（例如 `-m Curie：ft-<org>-<date>`）。其他训练参数无需更改，但是如果新的训练数据比以前的训练数据小得多，您可能会发现将 `learning_rate_multiplier` 降低 2 到 4 倍很有用。

# 权重和偏差

您可以将您的微调和[权重和偏差](https://wandb.me/openai-docs)进行同步以跟踪实验、模型和数据集。

要开始，您将需要一个[权重和偏差](https://wandb.me/openai-docs)帐户和一个付费的 OpenAI 计划。为了确保您正在使用最新版本的 `OpenAI` 和 `Wandb`，运行:

```bash
pip install --upgrade openai wandb
```

要将您的微调与 Weights＆Biases 同步，运行：

```bash
openai wandb sync
```

您可以阅读 Weights＆Biases 文档以获取有关此集成的更多信息。

# 示例笔记本

## 分类

<a href="https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb" target="_blank" rel="noreferrer" class="tag-link">finetuning-classification.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a>

这个笔记本将演示如何微调模型，以分类输入文本是否与棒球或曲棍球有关。我们将在笔记本中完成以下四个步骤：

1. **数据探索**将概述数据源及示例的内容。
2. **数据准备**将把我们的数据源转换成可用于微调的 jsonl 文件。
3. **微调**将启动微调作业并解释结果模型的性能。
4. **使用模型**将演示如何向微调的模型发送请求以获取预测。

## 问题回答

<div class="docs-tag-link-list"><a href="https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-1-collect-data.ipynb" target="_blank" rel="noreferrer" class="tag-link">olympics-1-collect-data.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a><a href="https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-2-create-qa.ipynb" target="_blank" rel="noreferrer" class="tag-link">olympics-2-create-qa.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a><a href="https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-3-train-qa.ipynb" target="_blank" rel="noreferrer" class="tag-link">olympics-3-train-qa.ipynb<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="tag-link-icon" width="1em" height="1em"><path fill-rule="evenodd" clip-rule="evenodd" d="M13.5859 5H11V3H17V9H15V6.4143L8.70718 12.7071L7.29297 11.2929L13.5859 5ZM9 5H5H3V7V15V17H5H13H15V15V11H13V15H5V7H9V5Z"></path></svg></a></div>

这个项目的想法是创建一个基于几段提供的文本的问答模型。基于 GPT-3 基础模型，当答案包含在段落中时，结果会很好，但是如果答案没有包含在段落中，基础模型往往仍然会尽力回答问题，常常导致虚构的答案。

为了创建一个只有在有足够背景的情况下才能回答问题的模型，我们首先创建了一个基于文本段落的问题和答案数据集。为了训练模型仅在答案存在时回答问题，我们还添加了对抗性示例，其中问题不匹配上下文。在这些情况下，我们要求模型输出“No sufficient context for answering the question”。

我们将在三个笔记本中完成此任务：

1. [第一个笔记本](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-1-collect-data.ipynb)集中于收集最近的数据，这些数据 GPT-3 未在预训练过程中看到。我们选择了 2020 年奥运会（实际上在 2021 年夏季举行），并下载了 713 个独特页面。我们通过单独的节将数据集组织起来，这将用作提问和回答的上下文。
2. [第二个笔记本](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-2-create-qa.ipynb)将利用 Davinci-instruct，在维基百科节的基础上提出一些问题，并根据该节回答这些问题。
3. [第三个笔记本](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-3-train-qa.ipynb)将利用上下文，问题和答案对的数据集，同时创建对抗性问题和上下文对，其中问题不是在那个上下文中生成的。在这些情况下，模型将被提示回答“No sufficient context for answering the question”。我们还将训练一个判别器模型，用于预测基于上下文是否可以回答问题。
