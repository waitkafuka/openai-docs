# 介绍

您可以通过任何编程语言通过 HTTP 请求与 API 进行交互，通过我们的官方 Python 绑定、官方 Node.js 库或社区维护的库。

要安装官方的 Python 绑定，请运行以下命令：

```bash
pip install openai
```

要安装官方的 Node.js 库，请在 Node.js 项目目录中运行以下命令：

```bash
npm install openai
```

## 身份验证

OpenAI API 使用 API 密钥进行身份验证。访问您的 [API 密钥页面](https://platform.openai.com/account/api-keys)以检索您将在请求中使用的 API 密钥。

请记住，您的 API 密钥是机密的！不要与他人共享，也不要在任何客户端代码（浏览器、应用程序）中公开它。生产请求必须通过您自己的后端服务器路由，其中可以从环境变量或密钥管理服务安全地加载您的 API 密钥。

所有 API 请求应该在 Authorization HTTP 头中包含您的 API 密钥，如下所示：

```
Authorization: Bearer OPENAI_API_KEY
```

## 请求组织

对于属于多个组织的用户，可以传递一个标题来指定哪个组织用于 API 请求。来自这些 API 请求的使用将计入指定组织的订阅配额。

示例 curl 命令：

```bash
curl https://api.openai.com/v1/models \
 -H "Authorization: Bearer $OPENAI_API_KEY" \
 -H "OpenAI-Organization: YOUR_ORG_ID"

```

`openai` Python 包的示例：

```python
import os
import openai
openai.organization = "YOUR_ORG_ID"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
```

`openai` Node.js 包的示例：

```javascript
import { Configuration, OpenAIApi } from 'openai';
const configuration = new Configuration({
  organization: 'YOUR_ORG_ID',
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const response = await openai.listEngines();
```

组织 ID 可以在您的[组织设置](https://platform.openai.com/account/org-settings)页面找到。

## 发起请求

您可以将以下命令粘贴到终端中以运行您的第一个 API 请求。请确保将`$OPENAI_API_KEY` 替换为您的秘密 API 密钥。

```bash
curl https://api.openai.com/v1/chat/completions \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer $OPENAI_API_KEY" \
 -d '{
"model": "gpt-3.5-turbo",
"messages": [{"role": "user", "content": "Say this is a test!"}],
"temperature": 0.7
}'
```

此请求查询 `gpt-3.5-turbo` 模型以完成以“Say this is a test”开头的文本。您应该收到一个类似于以下内容的响应：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo-0301",
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "\n\nThis is a test!"
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

现在，您已经生成了第一个聊天功能。我们可以看到 `finish_reason` 是 `stop`，这意味着 API 返回了模型生成的完整结果。在上面的请求中，我们只生成了一条消息，但您可以将 `n` 参数设置为生成多个消息以供选择。在此示例中，`gpt-3.5-turbo` 被用于[传统的文本完成](https://platform.openai.com/docs/guides/completion/introduction)任务。该模型也针对[聊天应用](https://platform.openai.com/docs/guides/chat)进行了优化。

## 模型

列出并描述 API 中可用的各种模型。您可以参考[模型文档](https://platform.openai.com/docs/models)来了解可用模型及其之间的区别。

### 列出模型

`GET https://api.openai.com/v1/models`

列出目前可用的模型列表，提供关于每个模型的基本信息，例如所有者和可用性。
:::code-group

```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

```python
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
```

```javascript
const { Configuration, OpenAIApi } = require('openai');
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const response = await openai.listModels();
```

:::
响应结果：

```json
{
  "data": [
    {
      "id": "model-id-0",
      "object": "model",
      "owned_by": "organization-owner",
      "permission": [...]
    },
    {
      "id": "model-id-1",
      "object": "model",
      "owned_by": "organization-owner",
      "permission": [...]
    },
    {
      "id": "model-id-2",
      "object": "model",
      "owned_by": "openai",
      "permission": [...]
    },
  ],
  "object": "list"
}
```

## 检索模型

`GET https://api.openai.com/v1/models/{model}`

检索模型实例，提供有关模型的基本信息，例如所有者和权限。

### path 参数

model <Badge text="string" type="info"/> <Badge text="required" type="danger" vertical="middle"/>  
用于此请求的模型的 ID。

:::code-group

```bash
curl https://api.openai.com/v1/models/text-davinci-003 \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

```python
import os

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.retrieve("text-davinci-003")

```

```javascript
const { Configuration, OpenAIApi } = require('openai');
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const response = await openai.retrieveModel('text-davinci-003');
```

## 文本补全

给定一个 prompt，模型将返回一个或者多个补全，并且还可以返回每个位置上备选令牌出现的概率。

### 创建一个补全请求

`POST https://api.openai.com/v1/completions`  
为提供的 prompt 和参数发起一个补全请求。

### request body

**model** <Badge text="string" type="plain"/> <Badge text="required" type="danger" vertical="middle"/>
模型 ID，你可以使用[模型列表](https://platform.openai.com/docs/api-reference/models/list)接口获取可用的模型列表，也可以查看[模型概述](https://platform.openai.com/docs/models/overview)查看它们的描述。

**prompt** <Badge text="string or array" type="plain"/> <Badge text="Optional  " type="plain"/> <Badge text="Defaults to <|endoftext|>" type="plain"/>
要生成完成命令的提示，其编码可以是字符串、字符串数组、令牌数组或令牌数组的数组。
请注意，`<|endoftext|>`是模型在训练过程中所看到的文档分隔符，因此如果没有指定提示，模型将生成一个新文档的开头。

**suffix** <Badge text="string" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to null" type="plain"/>  
插入文本完成后的后缀。
The suffix that comes after a completion of inserted text.

**max_tokens** <Badge text="integer" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 16" type="plain"/>  
要在完成中生成的最大[令牌](https://platform.openai.com/tokenizer)数。

你的提示标记数量加上最大标记数量不能超过模型的上下文长度。大多数模型的上下文长度为 2048 个标记（除了最新的支持 4096 个标记的模型）。

**temperature** <Badge text="number" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 1" type="plain"/>

应该使用哪个采样温度，在 0 和 2 之间选择。较高的值（如 0.8）会使输出更随机，而较低的值（如 0.2）会使其更加集中和确定性。

我们通常建议在它和 `top_p` 之间改变其中一个参数而不是两个都改变。

**top_p** <Badge text="number" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 1" type="plain"/>
一种与温度抽样不同的替代方法，叫做：核心抽样，其中模型考虑具有 top_p 概率质量的标记的结果。因此，0.1 意味着仅考虑组成前 10％概率质量的令牌。

我们通常建议修改它和 `temperature` 其中一种参数，而不是同时修改。

**n** <Badge text="integer" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 1" type="plain"/>

每个提示生成多少个回答。

注：由于此参数生成了许多回答，因此可能会快速消耗您的令牌配额。请谨慎使用，并确保您对`max_tokens`和`stop`设置了合理的值。

**stream** <Badge text="boolean" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to false" type="plain"/>

是否需要流式返回部分进度。如果设置，则令牌将作为数据的服务器发送事件发出，流以 `data: [DONE]` 消息终止。

**logprobs** <Badge text="integer" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to null" type="plain"/>
将最有可能的标记的对数概率，以及选择的标记包括在 `logprobs` 中。例如，如果 logprobs 为 5，则 API 将返回一个包含最有可能的 5 个标记的列表。API 始终返回抽样标记的 `logprob`，因此响应中可能有多达 `logprobs +1` 个元素。

`logprobs` 的最大值为 5。如果您需要更多，请通过我们的[帮助中心](https://help.openai.com/)与我们联系，并描述您的用例。

**echo** <Badge text="boolean" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to false" type="plain"/>

除了结果之外，把提示也回显出来。

**stop** <Badge text="string or array" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to null" type="plain"/>

最多 4 个序列，API 在这里将停止生成更多的标记。返回的文本不会包含停止序列。
(Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.)

**presence_penalty** <Badge text="number" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 0" type="plain"/>

-2.0 和 2.0 之间的数字。正值会根据新标记是否出现在现有文本中来惩罚，增加模型谈论新话题的可能性。

[更多关于频率和存在惩罚的信息，请参见更多信息。](https://platform.openai.com/docs/api-reference/parameter-details)

**frequency_penalty** <Badge text="number" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 0" type="plain"/>  
介于-2.0 和 2.0 之间的数字。正值根据新标记在迄今为止的文本中的现有频率对其进行惩罚，从而降低模型重复完全相同行的可能性。

[更多关于频率和存在惩罚的信息，请参见更多信息。](https://platform.openai.com/docs/api-reference/parameter-details)

**best_of** <Badge text="integer" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to 1" type="plain"/>  
生成的 `best_of` 个最好的结果，并且返回最好的那一个，并返回“最好”（每个标记的对数概率最高的）的那一个。结果不能流式传输。

当与 `n` 一起使用时，`best_of` 控制候选完整性的数量，`n` 指定要返回多少个– `best_of` 必须大于 `n`。

注意：由于该参数生成许多完整性，因此可能会很快消耗您的标记配额。使用时要谨慎，并确保您对 `max_tokens` 和 `stop` 设置合理。

**logit_bias** <Badge text="map" type="plain"/> <Badge text="Optional" type="plain"/> <Badge text="Defaults to null" type="plain"/>  
更改指定令牌在结果中出现的概率。

接受一个 json 对象，将令牌（使用 GPT 分词器中的令牌 ID 指定）映射到从-100 到 100 的相关偏置值。您可以使用此[分词器工具](https://platform.openai.com/tokenizer?view=bpe)（适用于 GPT-2 和 GPT-3）将文本转换为令牌 ID。在数学上，在抽样之前将偏置添加到模型生成的逻辑中。具体的影响取决于模型，但在-1 和 1 之间的值应该会减少或增加选择的可能性；如-100 或 100 的值应该会导致有关令牌的禁止或专属选择。

例如，您可以传递`{"50256":-100}`来防止生成 <|endoftext|> 这样的标记.

**user** <Badge text="string" type="plain"/> <Badge text="Optional" type="plain"/>  
一种唯一的标识符，代表您的终端用户，可以帮助 OpenAI 监控和检测滥用。[了解更多](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)。

（未完待续，原文地址：https://platform.openai.com/docs/api-reference/chat/create?lang=node.js）
