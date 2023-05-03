# 资源库

## python 库

我们提供了一个 Python 资源库，你可以如下安装

```bash
pip install openai
```

一旦安装完成，您可以使用这些库和您的密钥运行以下代码：

```python
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)
```

这些库也会安装一个命令行工具，您可以像下面这样使用：

```bash
openai api completions.create -m text-davinci-003 -p "Say this is a test" -t 0 -M 7 --stream
```

## Node.js 库

我们还提供了一个 Node.js 库，您可以通过在 Node.js 项目目录中运行以下命令来安装：

```bash
npm install openai
```

一旦安装完成，您可以使用该库和您的密钥运行以下代码：

```javascript
const { Configuration, OpenAIApi } = require('openai');
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const response = await openai.createCompletion({
  model: 'text-davinci-003',
  prompt: 'Say this is a test',
  temperature: 0,
  max_tokens: 7,
});
```

## 社区库

下面的库由开发者社区构建和维护。如果您想在此处添加新的库，请按照我们的[帮助中心](https://help.openai.com/en/articles/6684216-adding-your-api-client-to-the-community-libraries-page)中添加社区库的说明操作。

请注意，OpenAI 不验证这些项目的正确性或安全性。使用它们要**自行承担风险**！

### C# / .NET

<ul><li><a href="https://github.com/betalgo/openai" target="_blank" rel="noopener noreferrer">Betalgo.OpenAI</a> by <a href="https://github.com/betalgo" target="_blank" rel="noopener noreferrer">Betalgo</a></li><li><a href="https://github.com/OkGoDoIt/OpenAI-API-dotnet" target="_blank" rel="noopener noreferrer">OpenAI-API-dotnet</a> by <a href="https://github.com/OkGoDoIt" target="_blank" rel="noopener noreferrer">OkGoDoIt</a></li><li><a href="https://github.com/RageAgainstThePixel/OpenAI-DotNet" target="_blank" rel="noopener noreferrer">OpenAI-DotNet</a> by <a href="https://github.com/RageAgainstThePixel" target="_blank" rel="noopener noreferrer">RageAgainstThePixel</a></li></ul>

### C++

<ul><li><a href="https://github.com/D7EAD/liboai" target="_blank" rel="noopener noreferrer">liboai</a> by <a href="https://github.com/D7EAD" target="_blank" rel="noopener noreferrer">D7EAD</a></li></ul>

### Clojure

<ul><li><a href="https://github.com/wkok/openai-clojure" target="_blank" rel="noopener noreferrer">openai-clojure</a> by <a href="https://github.com/wkok" target="_blank" rel="noopener noreferrer">wkok</a></li></ul>

### Crystal

<ul><li><a href="https://github.com/sferik/openai-crystal" target="_blank" rel="noopener noreferrer">openai-crystal</a> by <a href="https://github.com/sferik" target="_blank" rel="noopener noreferrer">sferik</a></li></ul>

### Dart/Flutter

<ul><li><a href="https://github.com/anasfik/openai" target="_blank" rel="noopener noreferrer">openai</a> by <a href="https://github.com/anasfik" target="_blank" rel="noopener noreferrer">anasfik</a></li></ul>

### Delphi

<ul><li><a href="https://github.com/HemulGM/DelphiOpenAI" target="_blank" rel="noopener noreferrer">DelphiOpenAI</a> by <a href="https://github.com/HemulGM" target="_blank" rel="noopener noreferrer">HemulGM</a></li></ul>

### Elixir

<ul><li><a href="https://github.com/mgallo/openai.ex" target="_blank" rel="noopener noreferrer">openai.ex</a> by <a href="https://github.com/mgallo" target="_blank" rel="noopener noreferrer">mgallo</a></li></ul>

### Go

<ul><li><a href="https://github.com/sashabaranov/go-gpt3" target="_blank" rel="noopener noreferrer">go-gpt3</a> by <a href="https://github.com/sashabaranov" target="_blank" rel="noopener noreferrer">sashabaranov</a></li></ul>

### Java

<ul><li><a href="https://github.com/TheoKanning/openai-java" target="_blank" rel="noopener noreferrer">openai-java</a> by <a href="https://github.com/TheoKanning" target="_blank" rel="noopener noreferrer">Theo Kanning</a></li></ul>

### Julia

<ul><li><a href="https://github.com/rory-linehan/OpenAI.jl" target="_blank" rel="noopener noreferrer">OpenAI.jl</a> by <a href="https://github.com/rory-linehan" target="_blank" rel="noopener noreferrer">rory-linehan</a></li></ul>

### Kotlin

<ul><li><a href="https://github.com/Aallam/openai-kotlin" target="_blank" rel="noopener noreferrer">openai-kotlin</a> by <a href="https://github.com/Aallam" target="_blank" rel="noopener noreferrer">Mouaad Aallam</a></li></ul>

### Node.js

<ul><li><a href="https://www.npmjs.com/package/openai-api" target="_blank" rel="noopener noreferrer">openai-api</a> by <a href="https://github.com/Njerschow" target="_blank" rel="noopener noreferrer">Njerschow</a></li><li><a href="https://www.npmjs.com/package/openai-api-node" target="_blank" rel="noopener noreferrer">openai-api-node</a> by <a href="https://github.com/erlapso" target="_blank" rel="noopener noreferrer">erlapso</a></li><li><a href="https://www.npmjs.com/package/gpt-x" target="_blank" rel="noopener noreferrer">gpt-x</a> by <a href="https://github.com/ceifa" target="_blank" rel="noopener noreferrer">ceifa</a></li><li><a href="https://www.npmjs.com/package/gpt3" target="_blank" rel="noopener noreferrer">gpt3</a> by <a href="https://github.com/poteat" target="_blank" rel="noopener noreferrer">poteat</a></li><li><a href="https://www.npmjs.com/package/gpts" target="_blank" rel="noopener noreferrer">gpts</a> by <a href="https://github.com/thencc" target="_blank" rel="noopener noreferrer">thencc</a></li><li><a href="https://www.npmjs.com/package/@dalenguyen/openai" target="_blank" rel="noopener noreferrer">@dalenguyen/openai</a> by <a href="https://github.com/dalenguyen" target="_blank" rel="noopener noreferrer">dalenguyen</a></li><li><a href="https://github.com/tectalichq/public-openai-client-js" target="_blank" rel="noopener noreferrer">tectalic/openai</a> by <a href="https://tectalic.com/" target="_blank" rel="noopener noreferrer">tectalic</a></li></ul>

### PHP

<ul><li><a href="https://packagist.org/packages/orhanerday/open-ai" target="_blank" rel="noopener noreferrer">orhanerday/open-ai</a> by <a href="https://github.com/orhanerday" target="_blank" rel="noopener noreferrer">orhanerday</a></li><li><a href="https://github.com/tectalichq/public-openai-client-php" target="_blank" rel="noopener noreferrer">tectalic/openai</a> by <a href="https://tectalic.com/" target="_blank" rel="noopener noreferrer">tectalic</a></li><li><a href="https://github.com/openai-php/client" target="_blank" rel="noopener noreferrer">openai-php clinet</a> by <a href="https://github.com/openai-php" target="_blank" rel="noopener noreferrer">openai-php</a></li></ul>

### Python

<ul><li><a href="https://github.com/OthersideAI/chronology" target="_blank" rel="noopener noreferrer">chronology</a> by <a href="https://www.othersideai.com/" target="_blank" rel="noopener noreferrer">OthersideAI</a></li></ul>

### R

<ul><li><a href="https://github.com/ben-aaron188/rgpt3" target="_blank" rel="noopener noreferrer">rgpt3</a> by <a href="https://github.com/ben-aaron188" target="_blank" rel="noopener noreferrer">ben-aaron188</a></li></ul>

### Ruby

<ul><li><a href="https://github.com/nileshtrivedi/openai/" target="_blank" rel="noopener noreferrer">openai</a> by <a href="https://github.com/nileshtrivedi" target="_blank" rel="noopener noreferrer">nileshtrivedi</a></li><li><a href="https://github.com/alexrudall/ruby-openai" target="_blank" rel="noopener noreferrer">ruby-openai</a> by <a href="https://github.com/alexrudall" target="_blank" rel="noopener noreferrer">alexrudall</a></li></ul>

### Rust

<ul><li><a href="https://github.com/64bit/async-openai" target="_blank" rel="noopener noreferrer">async-openai</a> by <a href="https://github.com/64bit" target="_blank" rel="noopener noreferrer">64bit</a></li><li><a href="https://github.com/lbkolev/fieri" target="_blank" rel="noopener noreferrer">fieri</a> by <a href="https://github.com/lbkolev" target="_blank" rel="noopener noreferrer">lbkolev</a></li></ul>

### Scala

<ul><li><a href="https://github.com/cequence-io/openai-scala-client" target="_blank" rel="noopener noreferrer">openai-scala-client</a> by <a href="https://github.com/cequence-io" target="_blank" rel="noopener noreferrer">cequence-io</a></li></ul>

### Swift

<ul><li><a href="https://github.com/dylanshine/openai-kit" target="_blank" rel="noopener noreferrer">OpenAIKit</a> by <a href="https://github.com/dylanshine" target="_blank" rel="noopener noreferrer">dylanshine</a></li></ul>

### Unity

<ul><li><a href="https://github.com/hexthedev/OpenAi-Api-Unity" target="_blank" rel="noopener noreferrer">OpenAi-Api-Unity</a> by <a href="https://github.com/hexthedev" target="_blank" rel="noopener noreferrer">hexthedev</a></li><li><a href="https://github.com/RageAgainstThePixel/com.openai.unity" target="_blank" rel="noopener noreferrer">com.openai.unity</a> by <a href="https://github.com/RageAgainstThePixel" target="_blank" rel="noopener noreferrer">RageAgainstThePixel</a></li></ul>

### Unreal Engine

<ul><li><a href="https://github.com/KellanM/OpenAI-Api-Unreal" target="_blank" rel="noopener noreferrer">OpenAI-Api-Unreal</a> by <a href="https://github.com/KellanM" target="_blank" rel="noopener noreferrer">KellanM</a></li></ul>
