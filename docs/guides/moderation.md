# 内容分级

## 概述

Moderation 端点是一个工具，您可以使用它来检查内容是否符合 OpenAI 的使用政策。开发人员可以识别违反我们使用政策的内容，并采取行动，例如过滤掉它。

该模型将分类以下类别：

| 类别        | 描述                                                                                   |
| ----------- | -------------------------------------------------------------------------------------- |
| 仇恨        | 表达、煽动或宣传基于种族、性别、族裔、宗教、国籍、性取向、残疾状况或种姓的仇恨的内容。 |
| 仇恨/威胁   | 包括针对目标群体的暴力或严重伤害的仇恨内容。                                           |
| 自残        | 促进、鼓励或描绘自残行为（如自杀、自残和饮食障碍）的内容。                             |
| 性          | 旨在激起性欲的内容，如性行为描述或宣传性服务（不包括性教育和健康）。                   |
| 性/未成年人 | 包括未满 18 岁的个人的性内容。                                                         |
| 暴力        | 宣传或美化暴力或庆祝他人的痛苦或羞辱的内容。                                           |
| 暴力/图像   | 极其详细地描述死亡、暴力或严重身体伤害的暴力内容。                                     |

在监控 OpenAI API 的输入和输出时，Moderation 端点是免费使用的。我们目前不支持监控第三方流量。

:::info 提示
我们不断努力提高分类器的准确性，特别是对仇恨、自残和暴力/图像内容的分类。我们对非英语语言的支持目前还比较有限。
:::

## 快速入门

要获取文本的分级，可以像以下代码片段中所演示的那样向 [Moderation 端点](https://platform.openai.com/docs/api-reference/moderations)发出请求：

:::code-group

```bash
curl https://api.openai.com/v1/moderations \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"input": "Sample text goes here"}'
```

```python
response = openai.Moderation.create(
    input="Sample text goes here"
)
output = response["results"][0]
```

:::
以下是端点的示例输出。它返回以下字段：

- `flagged`：如果模型将内容分类为违反 OpenAI 使用政策，则设置为 true；否则为 false。
- `categories`：包含每个类别的二进制使用政策违规标志的字典。对于每个类别，如果模型将相应类别标记为违规，则该值为 true，否则为 false。
- `category_scores`：包含模型输出的每个类别的原始得分字典，表示模型对输入是否违反 OpenAI 该类别的政策的自信程度。该值介于 0 和 1 之间，其中较高的值表示较高的置信度。这些得分不应被解释为概率。

```json
{
  "id": "modr-XXXXX",
  "model": "text-moderation-001",
  "results": [
    {
      "categories": {
        "hate": false,
        "hate/threatening": false,
        "self-harm": false,
        "sexual": false,
        "sexual/minors": false,
        "violence": false,
        "violence/graphic": false
      },
      "category_scores": {
        "hate": 0.18805529177188873,
        "hate/threatening": 0.0001250059431185946,
        "self-harm": 0.0003706029092427343,
        "sexual": 0.0008735615410842001,
        "sexual/minors": 0.0007470346172340214,
        "violence": 0.0041268812492489815,
        "violence/graphic": 0.00023186142789199948
      },
      "flagged": false
    }
  ]
}
```

:::info 提示
OpenAI 将不断升级 Moderation 端点的底层模型。因此，依赖 `category_scores` 的自定义政策可能需要随时间重新校准。
:::
