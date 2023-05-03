# 聊天功能

利用聊天补全 API，您可以使用 `gpt-3.5-turbo` 和 `gpt-4` 构建自己的应用程序，它可以做以下事情：

- 起草电子邮件或其他写作
- 编写 Python 代码
- 回答有关一组文件的问题
- 创建对话客户端
- 为您的软件提供自然语言接口
- 辅导各种学科
- 翻译语言
- 为视频游戏模拟角色等等
  本指南介绍如何[使用 API 调用基于聊天的语言模型](https://platform.openai.com/docs/api-reference/chat)，并分享获取良好结果的技巧。您还可以在 [OpenAI Playground](https://platform.openai.com/playground?mode=chat) 中尝试使用新的聊天格式。

## 介绍

聊天模型将一系列消息作为输入，并将模型生成的消息作为输出返回。

尽管聊天格式旨在使多轮对话易于进行，但它对没有任何对话的单轮任务同样有用（例如以前由指令跟随模型如 `text-davinci-003` 提供的服务）。

示例 API 调用如下所示：

```python
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
```

主要的输入是消息参数。消息必须是消息对象的数组，其中每个对象具有角色（“系统”，“用户”或“助手”）和内容（消息的内容）。对话可以短到 1 个消息或填满许多页面。

通常，对话以系统消息开头，然后是交替的用户和助手消息。

系统消息有助于设置助手的行为。在上面的示例中，助手被指示为“您是一个有用的助手”。

:::tip 提示
gpt-3.5-turbo-0301 并不总是强烈关注系统消息，未来的模型将被训练以更强烈地关注系统消息。
:::

用户消息有助于指导助手。它们可以由应用程序的最终用户生成，也可以作为指令由开发人员设定。

助手消息有助于存储先前的响应。它们也可以由开发人员编写，以帮助提供所需行为的示例。

包含对话历史记录有助于当用户指令涉及先前的消息时，系统进行回复。在上面的示例中，用户的最后一个问题“它在哪里进行？”只有在关于 2020 年世界系列赛的先前消息上下文中才有意义。因为模型没有记忆先前的请求，所有相关信息必须通过对话提供。如果对话无法适应模型的令牌限制，它必须以某种方式缩短。

## 响应格式

示例 API 响应如下所示：

```python
{
 'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-3.5-turbo',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': 'The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers.'},
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}
```

在 Python 中，助手的回应可以使用 `response['choices'][0]['message']['content']`获取。

每个响应都将包括一个 finish_reason。finish_reason 的可能值为：

- `stop`：API 返回完整的模型输出
- `length`：由于 [max_tokens 参数](https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens)或令牌限制导致不完整的模型输出
- `content_filter`：由于内容过滤器的标志而省略的内容
- `null`：API 响应仍在进行或不完整

## 管理令牌（token）

语言模型以被称为令牌的块读取文本。在英语中，令牌可以短至一个字符或长至一个单词（例如，a 或 apple），在某些语言中，令牌甚至可以比一个字符更短或比一个单词更长。

例如，`“ChatGPT is great！”`字符串被编码为六个令牌：`[“Chat”，“G”，“PT”，“ is”，“ great”，“！”]`。

API 调用中令牌的总数影响以下几个方面：

- 您的 API 调用成本，因为按令牌计费
- 您的 API 调用需要的时间，因为编码更多令牌需要更多时间
- 您的 API 调用是否起作用，因为总令牌必须低于模型的最大限制（`gpt-3.5-turbo-0301` 的 4096 个令牌）

输入和输出令牌都计入这些数量。例如，如果您的 API 调用在消息输入中使用了 10 个令牌，并且您在消息输出中收到了 20 个令牌，则将为您计费 30 个令牌。

要查看 API 调用使用了多少令牌，请检查 API 响应中的 `usage` 字段（例如，`response['usage']['total_tokens']`）。

像 `gpt-3.5-turbo` 和 `gpt-4` 这样的聊天模型以与其他模型相同的方式使用令牌，但由于其基于消息的格式，对令牌数量的计算更加困难。

:::details 深入挖掘 token 计算
以下是一个用于计算传递到 `gpt-3.5-turbo-0301` 的消息标记的函数示例。
消息转换为令牌的确切方式可能因模型而异。因此，当未来的模型版本发布时，此功能返回的答案可能只是近似值。 [ChatML 文档](https://github.com/openai/openai-python/blob/main/chatml.md)解释了 OpenAI API 如何将消息转换为 token，并且可能有助于编写您自己的函数。

```python
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
```

接下来，创建一条消息并将其传递给上面定义的函数，以查看令牌计数，这应该与 API 返回的值匹配。

```python
messages = [
  {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
  {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
  {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
  {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
  {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
  {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
]

model = "gpt-3.5-turbo-0301"

print(f"{num_tokens_from_messages(messages, model)} prompt tokens counted.")
# Should show ~126 total_tokens
```

为了确认上述函数生成的数字与 API 返回的数字相同，请创建一个新的聊天接口调用：

```python
# example token count from the OpenAI API
import openai


response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,
)

print(f'{response["usage"]["prompt_tokens"]} prompt tokens used.')
```

:::

要在不进行 API 调用的情况下查看文本字符串中有多少个令牌，请使用 OpenAI 的 [tiktoken](https://github.com/openai/tiktoken) Python 库。示例代码可以在 OpenAI Cookbook 的 [tiktoken 计数令牌的指南](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)中找到。

传递给 API 的每条消息都会消耗 token，包括内容、角色和其他字段，以及一些额外的隐藏格式。这些未来可能会有变化。

如果对话具有太多的令牌，超出了模型的最大限制（例如，`gpt-3.5-turbo` 的 4096 个以上的令牌），则必须将您的文本截断，省略或缩小，直到其符合大小。请注意，如果从消息输入中删除了消息，则模型将失去所有有关它的知识。

还要注意，非常长的对话更有可能收到不完整的回复。例如，一个长度为 4090 个令牌的 `gpt-3.5-turbo` 对话在仅 6 个令牌后就被截断了。

## 调教模型

调教模型的最佳实践可能会随着模型版本升级而改变。以下的建议适用于 `gpt-3.5-turbo-0301`，可能不适用于未来的模型。

许多对话以一个系统消息绅士地指导助手开始。例如，这是 ChatGPT 使用的其中一条系统消息:

:::tip prompt
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}
:::

一般来说，`gpt-3.5-turbo-0301` 不会过分关注系统消息，因此重要的指令通常最好放在用户消息中。

如果模型没有生成您想要的输出，请随时迭代并尝试改进。您可以尝试如下方法：

- 使您的指导更加明确
- 指定您希望答案的格式
- 要求模型在做出答案之前逐步思考或辩论利弊

有关更多提示工程化的想法，请阅读 [OpenAI cookbook 指南](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)，了解如何提高可靠性的技术。

除了系统消息之外，`temperature` 和 `max_tokens` 是开发人员有很多选项来[影响聊天模型输出中的两个选项](https://platform.openai.com/docs/api-reference/chat)。对于 `temperature`，像 0.8 这样更高的值会使输出更加随机，而像 0.2 这样更低的值会使其更加专注和确定性。在 `max_tokens` 的情况下，如果您想将响应限制在一定长度范围内，最大标记可以设置为任意数字。例如，如果您将最大标记值设置为 5，则输出将被截断，结果对用户来说是不合理的。

## Chat VS Completion

由于 `gpt-3.5-turbo` 的性能与 `text-davinci-003` 相似，但每个标记的价格只有其 10％，因此我们建议在大多数用例中使用 `gpt-3.5-turbo`。

对于许多开发人员，转换就像简单地重新编写和重新测试 prompt 一样简单。

例如，如果您使用以下完成提示将英语翻译为法语：

```
将以下英语文本翻译为法语：“{text}”
```

那么以 Chat API 来调用就像下面这样：

```python
[
{"role"：“system”，“content”：“您是一个有用的助手，可以将英语翻译成法语。”}，
{"role"：“user”，“content”：'将以下英语文本翻译为法语：“{text}”'}
]
```

甚至只是用户消息：

```
[
{"role"：“user”，“content”：'将以下英语文本翻译为法语：“{text}”}
]
```

## 常见问题

### `gpt-3.5-turbo` 可以进行微调吗？

不。截至 2023 年 3 月 1 日，您只能对基本 `GPT-3` 模型进行微调。有关如何使用微调模型的更多详细信息，请参见[微调指南](https://platform.openai.com/docs/guides/fine-tuning)。

### 你们是否存储通过 API 传递的数据？

截至 2023 年 3 月 1 日，我们保留 API 数据 30 天，但不再使用通过 API 发送的数据来改进我们的模型。请在我们的[数据使用政策](https://openai.com/policies/usage-policies)中了解更多。

### 添加一个内容过滤层

如果您想将内容审查层添加到 Chat API 的输出中，您可以按照我们的[内容审查指南](../models#审核)来防止显示违反 OpenAI 使用政策的内容。
