# 模型

## 概览

OpenAI API 提供不同价格和能力的多种模型。您还可以通过[微调](https://platform.openai.com/docs/guides/fine-tuning)对我们的原始基础模型进行有限的自定义，以适应您的特定使用情况。

<div class="docs-models-toc"><table><thead><tr><th>模型</th><th>描述</th></tr></thead><tbody><tr><td><a href="/docs/models/gpt-4">GPT-4</a> <div class="css-1chnrrf">Limited beta</div></td><td>一组比 GPT-3.5 更好的模型，能够理解并生成自然语言或代码</td></tr><tr><td><a href="/docs/models/gpt-3-5">GPT-3.5</a></td><td>一组比 GPT-3 更好的模型，能够理解并生成自然语言或代码</td></tr><tr><td><a href="/docs/models/dall-e">DALL·E</a><div class="css-1chnrrf">Beta</div></td><td>一种可以根据自然语言提示生成和编辑图像的模型</td></tr><tr><td><a href="/docs/models/whisper">Whisper</a><div class="css-1chnrrf">Beta</div></td><td>一种可以将音频转换为文本的模型</td></tr><tr><td><a href="/docs/models/embeddings">Embeddings</a></td><td>一组可以将文本转换为数值类型的模型</td></tr><tr><td><a href="/docs/models/moderation">Moderation</a></td><td>一种经过细调的模型，能够检测文本是否可能敏感或不安全</td></tr><tr><td><a href="/docs/models/gpt-3">GPT-3</a></td><td>一组能理解和生成自然语言的模型</td></tr><tr><td><a href="/docs/models/codex">Codex</a><div class="css-1chnrrf">Deprecated</div></td><td>一组能理解和生成代码的模型，包括将自然语言翻译成代码</td></tr></tbody></table></div>
我们还发布了包括Point-E、Whisper、Jukebox和CLIP在内的开源模型。

访问我们的[模型索引](https://platform.openai.com/docs/model-index-for-researchers)以了解更多有关哪些模型在我们的研究论文中被提及以及 InstructGPT 和 GPT-3.5 等模型系列之间的区别。

## 模型和 API 端点兼容

<div class="models-table"><table><thead><tr><th>API 端点</th><th>模型名称</th></tr></thead><tbody><tr><td>/v1/chat/completions</td><td>gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301</td></tr><tr><td>/v1/completions</td><td>text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001</td></tr><tr><td>/v1/edits</td><td>text-davinci-edit-001, code-davinci-edit-001</td></tr><tr><td>/v1/audio/transcriptions</td><td>whisper-1</td></tr><tr><td>/v1/audio/translations</td><td>whisper-1</td></tr><tr><td>/v1/fine-tunes</td><td>davinci, curie, babbage, ada</td></tr><tr><td>/v1/embeddings</td><td>text-embedding-ada-002, text-search-ada-doc-001</td></tr><tr><td>/v1/moderations</td><td>text-moderation-stable, text-moderation-latest</td></tr></tbody></table></div>
这个列表不包括我们的第一代嵌入模型和我们的DALL·E模型。

## 后续的模型升级

随着 `gpt-3.5-turbo` 的发布，我们的一些模型现在正在不断更新。我们还提供静态模型版本，开发人员可以继续使用至少三个月，直到更新的模型推出。随着模型更新的新节奏，我们也给人们提供了评价的能力，以帮助我们改进不同用例的模型。如果您有兴趣，请查看 [OpenAI Evals](https://github.com/openai/evals) 存储库。

以下模型是临时快照，一旦更新版本可用，我们将宣布它们的废弃日期。如果您想使用最新的模型版本，请使用标准模型名称，如 `gpt-4` 或 `gpt-3.5-turbo`。

<table><thead><tr><th>Model name</th><th>Deprecation date</th></tr></thead><tbody><tr><td>gpt-3.5-turbo-0301</td><td>TBD 尚未确定（To be determined）</td></tr><tr><td>gpt-4-0314</td><td>TBD 尚未确定（To be determined）</td></tr><tr><td>gpt-4-32k-0314</td><td>TBD 尚未确定（To be determined）</td></tr></tbody></table>

## GPT-4 <Badge text="Limited beta" type="tip"/>

GPT-4 是一个大型多模态模型（接受文本输入和生成文本输出，未来将具有图像输入功能），可以比我们以前的任何模型更准确地解决困难问题，这得益于其更广泛的通用知识和先进的推理能力。与 `gpt-3.5-turbo` 一样，GPT-4 针对聊天进行了优化，但在传统的完成任务中也可以使用 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)。在我们的[聊天指南](https://platform.openai.com/docs/guides/chat)中学习如何使用 GPT-4。

:::tip 提示
GPT-4 目前处于有限的测试版阶段，只对获准访问权限的人可用。请[加入等待列表](https://openai.com/waitlist/gpt-4)，以便在有能力时获得访问权限。
:::

<div class="models-table"><table><thead><tr><th>最新模型</th><th>描述</th><th>最多的 token 支持</th><th>训练数据</th></tr></thead><tbody><tr><td>gpt-4</td><td>比任何GPT-3.5模型更有能力，能够执行更复杂的任务，并针对聊天进行了优化。将使用我们的最新模型迭代进行更新。</td><td>8,192 tokens</td><td>截止到 2021 年 9 月</td></tr><tr><td>gpt-4-0314</td><td>来自2023年3月14日的gpt-4快照。与gpt-4不同，该模型将不会接收更新，并将在发布新版本3个月后被弃用。</td><td>8,192 tokens</td><td>截止到 2021 年 9 月</td></tr><tr><td>gpt-4-32k</td><td>与基础gpt-4模型具有相同的功能，但具有4倍的上下文长度。将使用我们的最新模型迭代进行更新。</td><td>32,768 tokens</td><td>截止到 2021 年 9 月</td></tr><tr><td>gpt-4-32k-0314</td><td>来自2023年3月14日的gpt-4-32快照。与gpt-4-32k不同，该模型将不会接收更新，并将在发布新版本3个月后被弃用。</td><td>32,768 tokens</td><td>截止到 2021 年 9 月</td></tr></tbody></table></div>

## GPT-3.5

GPT-3.5 模型可以理解和生成自然语言或代码。我们在 GPT-3.5 系列中最具能力且成本效益最高的模型是 `gpt-3.5-turbo`，它已经针对聊天进行了优化，但也适用于传统的补全任务。

| 模型名称           | 说明                                                                                                                        | 最大标记 | 训练数据          |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------- | ----------------- |
| gpt-3.5-turbo      | 最具能力的 GPT-3.5 模型，优化聊天，成本为 text-davinci-003 的 1/10。将使用我们的最新模型迭代进行更新。                      | 4,096    | 截至 2021 年 9 月 |
| gpt-3.5-turbo-0301 | 来自 2023 年 3 月 1 日的 gpt-3.5-turbo 快照。与 gpt-3.5-turbo 不同，该模型将不会接收更新，并将在发布新版本 3 个月后被弃用。 | 4,096    | 截至 2021 年 9 月 |
| text-davinci-003   | 可以执行任何语言任务，质量更高，输出更长，且比 curie、babbage 或 ada 模型具有一致的指令遵循支持。还支持在文本中插入补全。   | 4,097    | 截至 2021 年 6 月 |
| text-davinci-002   | 与 text-davinci-003 具有类似的功能，但是使用监督微调而不是强化学习进行训练。                                                | 4,097    | 截至 2021 年 6 月 |
| code-davinci-002   | 专为代码补全任务进行了优化。                                                                                                | 8,001    | 截至 2021 年 6 月 |

我们建议使用 `gpt-3.5-turbo` 而不是其他 GPT-3.5 模型，因为它成本更低。

:::tip 注意
OpenAI 模型是不确定性的，这意味着相同的输入可能会产生不同的输出。将温度设置为 0 将使输出基本上成为确定性的，但可能仍然存在一小部分变异性。
:::

### 特定功能的模型

虽然新的 `gpt-3.5-turbo` 模型针对聊天进行了优化，但它在传统的补全任务中也非常有效。原始的 GPT-3.5 模型针对[文本补全](https://platform.openai.com/docs/guides/completion)进行了优化。

我们用于创建向量化和编辑文本的端点使用了它们自己的一组专业模型。

### 找到合适的模型

尝试使用 `gpt-3.5-turbo` 是了解 API 如何执行的好方法。在你有一个想要完成的任务的想法之后，可以使用 `gpt-3.5-turbo` 或另一个模型，并尝试围绕其能力进行优化。

您可以使用 [GPT 比较工具](https://gpttools.com/comparisontool)可并行运行不同的模型来比较输出、设置和响应时间，然后将数据下载到 Excel 电子表格中。

## DALL·E <Badge text="Beta" type="tip"/>

DALL·E 是一个能够从自然语言描述中创建逼真图像和艺术品的 AI 系统。我们目前支持根据提示创建新图像、编辑现有图像或创建用户提供图像的变体的能力。

当前 DALL·E 模型通过我们的 API 提供，是 DALL·E 的第二次迭代，比原始模型具有更真实、准确、分辨率更高 4 倍的图像。您可以通过我们的 [实验室界面](https://labs.openai.com/)或 API 进行尝试。

## Whisper <Badge text="Beta" type="tip"/>

Whisper 是一个通用语音识别模型。它训练于大量不同的音频数据集，并且是一个多任务模型，可以执行多语言语音识别以及语音翻译和语言识别。Whisper v2-large 模型目前可通过我们的 `whisper-1` 模型名称来使用。

目前，开源版本的 Whisper 与通过我们的 API 可用的版本没有区别。但是，通过我们的 API，我们提供了一种优化的推理过程，使通过我们的 API 运行 Whisper 比通过其他方式运行 Whisper 更快。有关 Whisper 的更多技术细节，您可以[阅读论文](https://openai.com/blog/new-and-improved-embedding-model)。

## 向量化

向量是文本的数值表示，可用于测量两个文本之间的相关性。我们的第二代向量化模型 `text-embedding-ada-002` 旨在以一小部分成本替换以前的 16 个第一代向量化模型。向量化于对于搜索、聚类、推荐、异常检测和分类任务非常有用。您可以在[公开论文](https://openai.com/blog/new-and-improved-embedding-model)中了解更多关于我们最新的向量化模型。

## 审核

审核模型旨在检查内容是否符合 OpenAI 的使用政策。该模型提供分类能力，查找以下类别的内容：仇恨、仇恨/威胁、自残、性、暴力图像。您可以在我们的[审查指南](https://platform.openai.com/docs/guides/moderation/overview)中了解更多信息。

审核模型接受任意大小的输入，自动分割以适应模型的特定上下文窗口。

| 模型                   | 描述                                       |
| ---------------------- | ------------------------------------------ |
| text-moderation-latest | 最强大的适度性模型。准确性略高于稳定模型。 |
| text-moderation-stable | 几乎和最新的模型一样强大，但略旧。         |

## GPT-3

GPT-3 模型可以理解和生成自然语言。
这些模型已被更强大的 GPT-3.5 代模型所取代。但是，原始的 GPT-3 基础模型（davinci、curie、ada 和 babbage）目前是唯一可用于微调的模型。

| 最新模型         | 描述                                                                 | 最大标记 | 培训数据           |
| ---------------- | -------------------------------------------------------------------- | -------- | ------------------ |
| text-curie-001   | 功能非常强大，速度比 Davinci 更快，成本更低。                        | 2,049    | 截至 2019 年 10 月 |
| text-babbage-001 | 能够完成简单的任务，非常快速，成本更低。                             | 2,049    | 截至 2019 年 10 月 |
| text-ada-001     | 能够完成非常简单的任务，通常是 GPT-3 系列中最快的模型，成本最低。    | 2,049    | 截至 2019 年 10 月 |
| davinci          | GPT-3 最强大的模型。可以完成其他模型能完成的任何任务，通常质量更高。 | 2,049    | 截至 2019 年 10 月 |
| curie            | 非常强大，但速度比 Davinci 更快，成本更低。                          | 2,049    | 截至 2019 年 10 月 |
| babbage          | 能够完成简单的任务，非常快速，成本更低。                             | 2,049    | 截至 2019 年 10 月 |
| ada              | 能够完成非常简单的任务，通常是 GPT-3 系列中最快的模型，成本最低。    | 2,049    | 截至 2019 年 10 月 |

## Codex <Badge text="Deprecated" type="tip"/>

Codex 模型现已停用。它们是我们的 GPT-3 模型的后代，可以理解和生成代码。它们的培训数据包含自然语言和 GitHub 中数十亿行公共代码。了解更多。

它们在 Python 中最强大，在 JavaScript、Go、Perl、PHP、Ruby、Swift、TypeScript、SQL 甚至 Shell 等十多种语言中也很娴熟。

以下 Codex 模型现已停用：

| 最新模型         | 描述                                                                                        | 最大标记           | 培训数据          |
| ---------------- | ------------------------------------------------------------------------------------------- | ------------------ | ----------------- |
| code-davinci-002 | 最强大的 Codex 模型。特别擅长将自然语言转化为代码。除了完成代码外，还支持在代码中插入完成。 | 8,001              | 截至 2021 年 6 月 |
| code-davinci-001 | code-davinci-002 的早期版本                                                                 | 8,001              | 截至 2021 年 6 月 |
| code-cushman-002 | 几乎和 Davinci Codex 一样强大，但速度略快。这种速度优势可能使其更适合实时应用程序。         | Up to 2,048 tokens |
| code-cushman-001 | code-cushman-002 的早期版本                                                                 | Up to 2,048 tokens |

更多信息，请参阅我们的 Codex [使用指南](https://platform.openai.com/docs/guides/code)。
