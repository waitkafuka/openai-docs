# 语音转文字 <Badge text="beta" type="tip"/>

学习如何将音频转换为文本

## 介绍

语音转文本 API 提供了基于我们最先进的开源大型 v2 [Whisper 模型](https://openai.com/blog/whisper/)的 `transcriptions` 和 `translations` 两个终点。它们可以用于：

- 将音频转录成音频所在的任何语言。
- 将音频翻译并转录成英语。

文件上传目前限制为 25 MB，支持以下输入文件类型：mp3，mp4，mpeg，mpga，m4a，wav 和 webm。

## 快速开始

### 转录

转录 API 将您要转录的音频文件和音频转录的所需输出文件格式作为输入。我们目前支持多种输入和输出文件格式。

:::code-group

```bash
curl --request POST \
  --url https://api.openai.com/v1/audio/transcriptions \
  --header 'Authorization: Bearer TOKEN' \
  --header 'Content-Type: multipart/form-data' \
  --form file=@/path/to/file/openai.mp3 \
  --form model=whisper-1
```

```python
# 注意：你需要使用 OpenAI Python v0.27.0 使下面的代码工作
import openai
audio_file= open("/path/to/file/audio.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
```

:::

默认情况下，响应类型将是 json，并包括原始文本。

```json
{
"text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger.
....
}
```

要在请求中设置其他参数，可以添加更多的 `--form` 行以及相关选项。例如，如果您想将输出格式设置为文本，可以添加以下行：

```
...
--form file=@openai.mp3 \
--form model=whisper-1 \
--form response_format=text
```

## 翻译

翻译 API 将任何支持的语言的音频文件作为输入，并在必要时将音频转录成英语。这与我们的/转录端点不同，因为输出不是原始输入语言而是被翻译成英语文本。

:::code-group

```python
# 注意：你需要使用 OpenAI Python v0.27.0 才能使下面的代码工作

import openai
audio_file = open("/path/to/file/german.mp3", "rb")
transcript = openai.Audio.translate("whisper-1", audio_file)
```

```bash
curl --request POST   --url https://api.openai.com/v1/audio/translations   --header 'Authorization: Bearer TOKEN'   --header 'Content-Type: multipart/form-data'   --form file=@/path/to/file/german.mp3   --form model=whisper-1
```

:::
在这种情况下，输入的音频是德语，输出的文本如下：

```
Hello, my name is Wolfgang and I come from Germany. Where are you heading today?
```

我们目前仅支持将翻译成英语。

## 支持的语言

我们目前通过转录和翻译终点[支持以下语言](https://github.com/openai/whisper#available-models-and-languages)：

南非荷兰语，阿拉伯语，亚美尼亚语，阿塞拜疆语，白俄罗斯语，波斯尼亚语，保加利亚语，加泰罗尼亚语，中文，克罗地亚语，捷克语，丹麦语，荷兰语，英语，爱沙尼亚语，芬兰语，法语，加利西亚语，德语，希腊语，希伯来语，印地语，匈牙利语，冰岛语，印度尼西亚语，意大利语，日语，卡纳达语，哈萨克语，韩语，拉脱维亚语，立陶宛语，马其顿语，马来语，马拉地语，毛利语，尼泊尔语，挪威语，波斯语，波兰语，葡萄牙语，罗马尼亚语，俄语，塞尔维亚语，斯洛伐克语，斯洛文尼亚语，西班牙语，斯瓦希里语，瑞典语，他加禄语，泰米尔语，泰语，土耳其语，乌克兰语，乌尔都语，越南语和威尔士语。

虽然基础模型是在 98 种语言上进行训练的，但我们只列出了超过<50％的单词[错误率](https://en.wikipedia.org/wiki/Word_error_rate)（WER），这是语音转文本模型准确性的行业标准基准。该模型将返回未列在上述语言中的语言的结果，但其质量将较低。

## 更长的输入

默认情况下，Whisper API 仅支持小于 25 MB 的文件。如果您有一个大于该文件长度的音频文件，则需要将其分成 25 MB 或更小的块或使用压缩的音频格式。为获得最佳性能，请避免在音频中间断断续续地断开音频，因为这可能会导致某些上下文丢失。

处理此问题的一种方法是使用 [PyDub 开源 Python 包](https://github.com/jiaaro/pydub)拆分音频：

```python
from pydub import AudioSegment

song = AudioSegment.from_mp3("good_morning.mp3")

# PyDub 以毫秒为时间单位

ten*minutes = 10 * 60 \_ 1000

first_10_minutes = song[:ten_minutes]

first_10_minutes.export("good_morning_10.mp3", format="mp3")
```

_OpenAI 不保证 PyDub 等第三方软件的可用性或安全性。_

## 提示

您可以使用 [prompt](https://platform.openai.com/docs/api-reference/audio/create#audio/create-prompt) 来提高 Whisper API 生成的转录质量。该模型将尝试匹配提示的样式，因此如果提示也使用大写字母和标点符号，它将更有可能使用它们。但是，当前的提示系统比我们的其他语言模型要受到限制，并且仅提供有限的对生成音频的控制。以下是提示在不同场景下如何帮助的一些示例：

1. 提示可以非常有帮助，用于更正模型在音频中经常错误识别的特定单词或缩写词。例如，以下提示改进了 DALL·E 和 GPT-3 的转录，之前将其写成“GDP 3”和“DALI”。

```
The transcript is about OpenAI which makes technology like DALL·E, GPT-3, and ChatGPT with the hope of one day building an AGI system that benefits all of humanity
```

2. 为了保留已分成段的文件的上下文，您可以使用上一段的转录提示模型。这将使转录更准确，因为模型将使用前一个音频的相关信息。模型仅考虑提示的最后 224 个令牌，忽略之前的任何内容。

3. 有时，模型在转录中可能会跳过标点符号。您可以使用一个包含标点符号的简单提示来避免出现这种情况：

```
Hello, welcome to my lecture.
```

4. 模型可能在音频中留下常见的充数词汇。如果您想在转录中保留填充词，请使用包含它们的提示：

```
Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
```

5. 某些语言可以用不同的方式书写，例如简体或繁体中文。该模型可能不始终使用您想要的书写风格进行转录。通过使用首选书写风格的提示来改善此情况。
