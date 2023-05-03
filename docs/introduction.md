# 概述

## 说明

OpenAI API 可应用于几乎任何涉及理解、生成自然语言，生成代码或图像的任务。我们提供一系列具有不同能力的模型，适用于不同的任务，同时具有微调自定义模型的能力。这些模型可用于内容生成、语义搜索、分类等各种用途。

## 关键概念

我们建议通过实践性的交互式示例完成快速入门教程，了解关键概念。

通过构建一个[快速示例应用](./quickstart)程序进行学习

### 提示词（Prompts）

Prompts 的设计思想，可以认为上是建立在如何对模型进行“编程”上。通常通过提供一些说明或几个示例来实现。这与大多数其他 NLP 服务不同，后者设计用于单个任务，例如情感分类或命名实体识别。恰恰相反，“completions”和“chat completions”终点可用于几乎任何通用任务，包括内容或代码生成、文本摘要、文本扩展、交谈、创意写作、格式转换等。

### 令牌（Tokens）

我们的模型通过将文本分解为”令牌“（Token）来理解和处理文本。令牌可以是单词，也可以是字符块。例如，“hamburger”这个词被分解成“ham”、“bur”和“ger”三个令牌，而像“pear”这样的短而常用的词是一个单一的令牌。许多令牌以空格开头，例如“ hello”和“ bye”。

在给定的 API 请求中，处理令牌的数量取决于你的输入和输出的长度。作为粗略的经验法则，对于英文文本，1 个 Token 约为 4 个字符或 0.75 个单词。一个限制是，你的文本提示和生成的组合必须不超过模型的最大上下文长度限制（对于大多数模型，这是 2048 个令牌，约为 1500 个单词）。查看我们的[令牌工具](https://platform.openai.com/tokenizer)，了解关于文本如何被转换为令牌的更多信息。

### 模型（Models）

API 使用一组具有不同能力和价格的模型提供支持。GPT-4 是我们最新且最强大的模型。GPT-3.5-Turbo 是 ChatGPT 的动力引擎，专为会话格式进行了优化。要了解有关这些模型以及我们提供的其他内容的更多信息，请访问我们的[模型文档](https://platform.openai.com/docs/models)。

## 下一步

- 在开始开发你的应用程序时，请牢记我们的[使用政策](https://openai.com/policies/usage-policies)。
- 浏览我们的[示例库](https://platform.openai.com/examples)，获得灵感。
- 参阅我们的指南之一，开始开发。

### 指南

[聊天 （Chat）](/guides/chat)  
学习如何使用基于聊天的语言模型

[文本补全（Text completion）](/guides/completion)  
学习如何生成和编辑文本

[向量化（Embeddings）](/guides/embeddings)  
学习如何搜索、分类和比较文本。

[语音转文字（Speech to text）](/guides/speech-to-text))  
学习如何把语音转为文字

[图片生成（Image generation）](/guides/images)  
学习如何生成和编辑图片

[模型微调（Fine-tuning）](/guides/fine-tuning)  
学习如何训练自己的模型
