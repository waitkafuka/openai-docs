# 快速开始

OpenAI 已经训练了先进的语言模型，它非常擅长理解和生成文本。我们的 API 提供对这些模型的访问，并可用于解决几乎任何语言相关的任务。

在这个快速入门教程中，你将构建一个简单的示例应用程序。在此过程中，你将学习到如何使用 API 进行文本处理任务所需的关键概念和技术，包括：

- 内容生成
- 内容摘要
- 内容编排、分类和情感分析
- 数据提取
- 翻译
- 其他任务

## 介绍

[文本补全](./example.md)端点是我们 API 的核心，提供了一个简单、灵活和强大的接口。你将一些文本作为**提示**输入，API 将返回一个文本**补全**，试图匹配你给定的任何指令或上下文。

<div class="completion-example-container"><div class="completion-example"><div class="completion-example-label body-small">Prompt</div><div>Write a tagline for an ice cream shop.</div><div></div><div class="completion-example-arrow"><svg style="display:inline-block" stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 20 20" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg></div><div class="completion-example-label body-small">Completion</div><div class="tutorial-example-completion">We serve up smiles with every scoop!</div></div></div>

你可以将其视为一种非常高级的自动完成，模型处理你的文本提示，并尝试预测接下来最可能出现的内容。

### 1. 从指令开始

想象一下你想开发一个“宠物名字生成器”。从头开始想出名字很难！

首先，你需要一个明确表明你想要什么的提示。让我们从一个指令开始。  
:::tip prompt
Suggest one name for a horse.
:::

不错！现在，试着让你的指令更加具体：  
:::tip prompt
Suggest one name for a black horse.
:::

正如你所看到的，将一个简单的形容词添加到我们的提示中会改变结果完成的内容。设计你的 prompt 本质上就是如何“编程”模型。

### 2. 给定一些例子

制定好的提示方案对于获得良好的结果非常重要，但有时这并不够。让我们尝试让你的说明更加复杂。

:::tip prompt
Suggest three names for a horse that is a superhero.
:::
这个补全的结果并不是我们想要的。这些名字相当普通，看来模型没有理解我们说明中的马的部分。让我们看看能否让它想出一些更相关的建议。

在许多情况下，向模型输入示例有助于传达模式或细微差别。尝试提交包含一些示例的提示。

:::tip prompt
Suggest three names for an animal that is a superhero.

Animal: Cat  
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline  
Animal: Dog  
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot  
Animal: Horse  
Names:  
:::

很好！通过给定我们期望输出的例子，帮助模型返回了我们正在寻找的名字类型。

### 3. 调整你的设置

提示词的构建并不是你手中的唯一工具。你还可以通过调整设置来控制程序如何进行补全。其中最重要的设置之一称为：“温度（temperature）”。

你可能已经注意到，如果在上面的示例中提交相同的提示多次，则该模型始终会返回相同或非常相似的完成。这是因为你的温度设置为 0。

尝试使用温度设置为 1 重新提交相同的提示几次。

:::tip prompt
为一只动物取三个超级英雄的名字。

动物：猫  
名字：尖爪队长、毛球特工、非凡的猫科动物  
动物：狗  
名字：保护者拉夫、奇迹犬、叫得多的爵士  
动物：马  
名字：  
:::

看看发生了什么？当温度高于 0 时，提交相同的 prompt 每次结果都不同。

请记住，模型预测哪些文本最有可能跟着之前的文本。温度是一个介于 0 和 1 之间的值，它本质上允许你控制模型在进行这些预测时应该有多大的“确定性”。降低温度意味着它将采取更少的风险，完成将更准确和更多的确定性。提高温度将导致更多样化的补全。

:::details 深入理解“标记（token）”和“可能性（probabilities）”
我们的模型通过将文本分解为称为 “标记（token）” 的较小单位来处理文本。标记可以是单词、单词块或单个字符。观察下面的文本是如何被标记化的。

<div class="escape-demo">

I have an orange cat named Butterscotch.

</div>

<div class="simple-tokenizer-output"><span class="tokenizer-tkn tokenizer-tkn-0">I</span><span class="tokenizer-tkn tokenizer-tkn-1"> have</span><span class="tokenizer-tkn tokenizer-tkn-2"> an</span><span class="tokenizer-tkn tokenizer-tkn-3"> orange</span><span class="tokenizer-tkn tokenizer-tkn-4"> cat</span><span class="tokenizer-tkn tokenizer-tkn-0"> named</span><span class="tokenizer-tkn tokenizer-tkn-1"> But</span><span class="tokenizer-tkn tokenizer-tkn-2">ters</span><span class="tokenizer-tkn tokenizer-tkn-3">cot</span><span class="tokenizer-tkn tokenizer-tkn-4">ch</span><span class="tokenizer-tkn tokenizer-tkn-0">.</span></div>
常见的像“猫”这样的单词被视为一个标记（token），而不常见的单词通常被拆分成多个标记。例如，“Butterscotch”可以翻译成四个标记：“But”、“ters”、“cot”和“ch”，很多标记以空格开头，例如“ hello”和“ bye”。

给定一些文本，模型确定哪个标记最有可能出现在下一个位置。例如，文本“Horses are my favorite”最有可能跟随的令牌是“animal”。

<div class="lpviz"><div><input class="text-input text-input-md text-input-full lpviz-input" type="text" maxlength="100" value="Horses are my favorite"></div><div class=""><div class="lpviz-logit"><div class="lpviz-logit-header"><div> animal</div><div class="lpviz-logit-percent">49.65%</div></div><div class="lpviz-logit-bar"><div class="lpviz-logit-bar-inner" style="transform: scaleX(0.496464);"></div></div></div><div class="lpviz-logit"><div class="lpviz-logit-header"><div> animals</div><div class="lpviz-logit-percent">42.58%</div></div><div class="lpviz-logit-bar"><div class="lpviz-logit-bar-inner" style="transform: scaleX(0.425821);"></div></div></div><div class="lpviz-logit"><div class="lpviz-logit-header"><div>\n</div><div class="lpviz-logit-percent">3.49%</div></div><div class="lpviz-logit-bar"><div class="lpviz-logit-bar-inner" style="transform: scaleX(0.0349244);"></div></div></div><div class="lpviz-logit"><div class="lpviz-logit-header"><div>!</div><div class="lpviz-logit-percent">0.91%</div></div><div class="lpviz-logit-bar"><div class="lpviz-logit-bar-inner" style="transform: scaleX(0.00905848);"></div></div></div></div></div>  
这就是“温度（temperature）”发挥作用的地方。如果你将温度设置为0，使用相同的提示进行4次提交，模型将始终返回“animal”，因为它具有最高的概率。如果你增加温度，则模型会冒更多风险，考虑概率较低的标记。
<div class="temp-example"><div class="temp-example-box"><div class="subheading">温度为 0 时：</div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animal</span></div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animal</span></div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animal</span></div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animal</span></div></div><div class="temp-example-box"><div class="subheading">温度为 1 时：</div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animal</span></div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animals</span></div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">!</span></div><div class="temp-example-sentence">Horses are my favorite <span class="tutorial-example-completion">animal</span></div></div></div>
通常情况下，在期望输出已经定义明确的任务中，最好将温度设置较低。而在期望多样性或创造性输出的任务中，更高的温度可能更适用，或者如果你希望生成一些变化供最终用户或人类专家选择，也可以采用更高的温度。
:::

对于你“宠物名称生成器”这个应用，你可能希望能够生成大量的名称。温度为
0.6 的中等范围应该很适合。

### 4. 构建你的应用程序

现在，你已经找到了一个好的提示和设置，准备好构建你的“宠物名称生成器”了！我们编写了一些代码来帮助你入门，请按照以下说明下载代码并运行应用程序。

##### 设置

如果你没有安装 Node.js，请从[此处安装](https://nodejs.org/en/)。然后通过克隆[此仓库](https://github.com/openai/openai-quickstart-node)下载代码。

```bash
git clone https://github.com/openai/openai-quickstart-node.git
```

如果你不想使用 git，则可以使用[此压缩文件](https://github.com/openai/openai-quickstart-node/archive/refs/heads/master.zip)另行下载代码。

##### 添加你的 API 密钥

要让应用程序正常工作，你需要一个 API 密钥。你可以通过[注册账户](https://platform.openai.com/signup)并返回此页面来获取密钥。

##### 运行应用程序

在项目目录中运行以下命令以安装依赖项并运行应用程序。

```bash
npm install
npm run dev
```

在浏览器中打开 <a href="http://localhost:3000" target="_blank" rel="noreferrer">http://localhost:3000</a>，你应该可以看到宠物名称生成器！

##### 理解代码

在 `openai-quickstart-node/pages/api` 文件夹中打开 `generate.js`。在底部，你将看到一个用来生成我们上面使用的 prompt 的函数。由于用户将输入其宠物的动物类型，因此它动态地改变了指定动物的那一部分 prompt。

```javascript
function generatePrompt(animal) {
  const capitalizedAnimal = animal[0].toUpperCase() + animal.slice(1).toLowerCase();
  return `Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: ${capitalizedAnimal}
Names:`;
}
```

在 `generate.js` 的第九行，你将看到发送实际 API 请求的代码。如上所述，它设定的温度（temperature）参数是 0.6。

```javascript
const completion = await openai.createCompletion({
  model: 'text-davinci-003',
  prompt: generatePrompt(req.body.animal),
  temperature: 0.6,
});
```

大功告成！现在，你已经完全了解了如何使用 OpenAI API 来构建你的“超级英雄宠物命名”程序。

## 计费

我们提供不同功能和[价格](https://openai.com/pricing/)的多种[模型](https://platform.openai.com/docs/models)以供选择。在本教程中，我们使用了 `text-davinci-003 `模型。我们建议在进行实验时使用该模型或 `gpt-3.5-turbo` 模型，因为它们将产生最佳结果。一旦它们正常工作，就可以看看其他模型是否可以以更低的延迟和成本产生相同的结果。或者您可能需要升级到更强大的模型，比如 `gpt-4`。

在单个请求中处理的标记总数（包括 prompt 和 completion）不能超过模型的最大上下文长度。对于大多数模型，这是 4,096 个令牌或者大约 3,000 个单词。粗略的估算是，一个令牌大约相当于英文文本的 4 个字符或者 0.75 个单词。

我们按每 1,000 个标记计费，并提供$5 的免费信用额度，可以在您的前 3 个月内使用。[了解更多信息](https://openai.com/pricing/)。

## 结语

这些概念和技术将有助于您构建自己的应用程序。但是，这个简单的例子只是展示了能够实现的一小部分！完成端点足够灵活，可以解决几乎所有的自然语言处理任务，包括内容生成、摘要、语义搜索、主题标记、情感分析等。

需要注意的一点限制是，对于大多数模型，单个 API 请求在提示和完成之间最多只能处理 4,096 个标记。

对于更复杂的任务，您可能需要提供更多的样例或上下文，超出了单个提示可以容纳的范围。[微调 API](https://platform.openai.com/docs/guides/fine-tuning) 是处理此类更高级任务的绝佳选择。微调允许您提供数百甚至数千个示例，以自定义模型来适应您特定的场景。

## 下一步

为了获取灵感来设计更多更好的 prompt，你可以：

- 阅读我们的[完成指南](https://platform.openai.com/docs/guides/completion)。
- 探索我们的[示例提示](https://platform.openai.com/examples)。
- 在“[Playground](https://platform.openai.com/playground)”进行实验。
- 在开始构建应用程序之前，请牢记我们的[使用政策](https://openai.com/policies/usage-policies)。
