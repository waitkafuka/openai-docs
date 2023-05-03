import{_ as a,c as e,o as n,V as r}from"./chunks/framework.61392f5d.js";const D=JSON.parse('{"title":"资源库","description":"","frontmatter":{},"headers":[],"relativePath":"libraries.md","filePath":"libraries.md","lastUpdated":null}'),s={name:"libraries.md"},l=r(`<h1 id="资源库" tabindex="-1">资源库 <a class="header-anchor" href="#资源库" aria-label="Permalink to &quot;资源库&quot;">​</a></h1><h2 id="python-库" tabindex="-1">python 库 <a class="header-anchor" href="#python-库" aria-label="Permalink to &quot;python 库&quot;">​</a></h2><p>我们提供了一个 Python 资源库，你可以如下安装</p><div class="language-bash line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#FFCB6B;">pip</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">install</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">openai</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><p>一旦安装完成，您可以使用这些库和您的密钥运行以下代码：</p><div class="language-python line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> os</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> openai</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># Load your API key from an environment variable or secret management service</span></span>
<span class="line"><span style="color:#A6ACCD;">openai</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">api_key</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> os</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">getenv</span><span style="color:#89DDFF;">(</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">OPENAI_API_KEY</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#A6ACCD;">response </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> openai</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">Completion</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">create</span><span style="color:#89DDFF;">(</span><span style="color:#A6ACCD;font-style:italic;">model</span><span style="color:#89DDFF;">=</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">text-davinci-003</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">prompt</span><span style="color:#89DDFF;">=</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">Say this is a test</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">temperature</span><span style="color:#89DDFF;">=</span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">max_tokens</span><span style="color:#89DDFF;">=</span><span style="color:#F78C6C;">7</span><span style="color:#89DDFF;">)</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br><span class="line-number">6</span><br><span class="line-number">7</span><br></div></div><p>这些库也会安装一个命令行工具，您可以像下面这样使用：</p><div class="language-bash line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#FFCB6B;">openai</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">api</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">completions.create</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">-m</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">text-davinci-003</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">-p</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">Say this is a test</span><span style="color:#89DDFF;">&quot;</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">-t</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">-M</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">7</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">--stream</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><h2 id="node-js-库" tabindex="-1">Node.js 库 <a class="header-anchor" href="#node-js-库" aria-label="Permalink to &quot;Node.js 库&quot;">​</a></h2><p>我们还提供了一个 Node.js 库，您可以通过在 Node.js 项目目录中运行以下命令来安装：</p><div class="language-bash line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#FFCB6B;">npm</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">install</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">openai</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><p>一旦安装完成，您可以使用该库和您的密钥运行以下代码：</p><div class="language-javascript line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">javascript</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#C792EA;">const</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">{</span><span style="color:#A6ACCD;"> Configuration</span><span style="color:#89DDFF;">,</span><span style="color:#A6ACCD;"> OpenAIApi </span><span style="color:#89DDFF;">}</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">require</span><span style="color:#A6ACCD;">(</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">openai</span><span style="color:#89DDFF;">&#39;</span><span style="color:#A6ACCD;">)</span><span style="color:#89DDFF;">;</span></span>
<span class="line"><span style="color:#C792EA;">const</span><span style="color:#A6ACCD;"> configuration </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">new</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">Configuration</span><span style="color:#A6ACCD;">(</span><span style="color:#89DDFF;">{</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#F07178;">apiKey</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> process</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">env</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">OPENAI_API_KEY</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#89DDFF;">}</span><span style="color:#A6ACCD;">)</span><span style="color:#89DDFF;">;</span></span>
<span class="line"><span style="color:#C792EA;">const</span><span style="color:#A6ACCD;"> openai </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">new</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">OpenAIApi</span><span style="color:#A6ACCD;">(configuration)</span><span style="color:#89DDFF;">;</span></span>
<span class="line"><span style="color:#C792EA;">const</span><span style="color:#A6ACCD;"> response </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;font-style:italic;">await</span><span style="color:#A6ACCD;"> openai</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">createCompletion</span><span style="color:#A6ACCD;">(</span><span style="color:#89DDFF;">{</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#F07178;">model</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">text-davinci-003</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#F07178;">prompt</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">Say this is a test</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#F07178;">temperature</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#F07178;">max_tokens</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">7</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#89DDFF;">}</span><span style="color:#A6ACCD;">)</span><span style="color:#89DDFF;">;</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br><span class="line-number">6</span><br><span class="line-number">7</span><br><span class="line-number">8</span><br><span class="line-number">9</span><br><span class="line-number">10</span><br><span class="line-number">11</span><br></div></div><h2 id="社区库" tabindex="-1">社区库 <a class="header-anchor" href="#社区库" aria-label="Permalink to &quot;社区库&quot;">​</a></h2><p>下面的库由开发者社区构建和维护。如果您想在此处添加新的库，请按照我们的<a href="https://help.openai.com/en/articles/6684216-adding-your-api-client-to-the-community-libraries-page" target="_blank" rel="noreferrer">帮助中心</a>中添加社区库的说明操作。</p><p>请注意，OpenAI 不验证这些项目的正确性或安全性。使用它们要<strong>自行承担风险</strong>！</p><h3 id="c-net" tabindex="-1">C# / .NET <a class="header-anchor" href="#c-net" aria-label="Permalink to &quot;C# / .NET&quot;">​</a></h3><ul><li><a href="https://github.com/betalgo/openai" target="_blank" rel="noopener noreferrer">Betalgo.OpenAI</a> by <a href="https://github.com/betalgo" target="_blank" rel="noopener noreferrer">Betalgo</a></li><li><a href="https://github.com/OkGoDoIt/OpenAI-API-dotnet" target="_blank" rel="noopener noreferrer">OpenAI-API-dotnet</a> by <a href="https://github.com/OkGoDoIt" target="_blank" rel="noopener noreferrer">OkGoDoIt</a></li><li><a href="https://github.com/RageAgainstThePixel/OpenAI-DotNet" target="_blank" rel="noopener noreferrer">OpenAI-DotNet</a> by <a href="https://github.com/RageAgainstThePixel" target="_blank" rel="noopener noreferrer">RageAgainstThePixel</a></li></ul><h3 id="c" tabindex="-1">C++ <a class="header-anchor" href="#c" aria-label="Permalink to &quot;C++&quot;">​</a></h3><ul><li><a href="https://github.com/D7EAD/liboai" target="_blank" rel="noopener noreferrer">liboai</a> by <a href="https://github.com/D7EAD" target="_blank" rel="noopener noreferrer">D7EAD</a></li></ul><h3 id="clojure" tabindex="-1">Clojure <a class="header-anchor" href="#clojure" aria-label="Permalink to &quot;Clojure&quot;">​</a></h3><ul><li><a href="https://github.com/wkok/openai-clojure" target="_blank" rel="noopener noreferrer">openai-clojure</a> by <a href="https://github.com/wkok" target="_blank" rel="noopener noreferrer">wkok</a></li></ul><h3 id="crystal" tabindex="-1">Crystal <a class="header-anchor" href="#crystal" aria-label="Permalink to &quot;Crystal&quot;">​</a></h3><ul><li><a href="https://github.com/sferik/openai-crystal" target="_blank" rel="noopener noreferrer">openai-crystal</a> by <a href="https://github.com/sferik" target="_blank" rel="noopener noreferrer">sferik</a></li></ul><h3 id="dart-flutter" tabindex="-1">Dart/Flutter <a class="header-anchor" href="#dart-flutter" aria-label="Permalink to &quot;Dart/Flutter&quot;">​</a></h3><ul><li><a href="https://github.com/anasfik/openai" target="_blank" rel="noopener noreferrer">openai</a> by <a href="https://github.com/anasfik" target="_blank" rel="noopener noreferrer">anasfik</a></li></ul><h3 id="delphi" tabindex="-1">Delphi <a class="header-anchor" href="#delphi" aria-label="Permalink to &quot;Delphi&quot;">​</a></h3><ul><li><a href="https://github.com/HemulGM/DelphiOpenAI" target="_blank" rel="noopener noreferrer">DelphiOpenAI</a> by <a href="https://github.com/HemulGM" target="_blank" rel="noopener noreferrer">HemulGM</a></li></ul><h3 id="elixir" tabindex="-1">Elixir <a class="header-anchor" href="#elixir" aria-label="Permalink to &quot;Elixir&quot;">​</a></h3><ul><li><a href="https://github.com/mgallo/openai.ex" target="_blank" rel="noopener noreferrer">openai.ex</a> by <a href="https://github.com/mgallo" target="_blank" rel="noopener noreferrer">mgallo</a></li></ul><h3 id="go" tabindex="-1">Go <a class="header-anchor" href="#go" aria-label="Permalink to &quot;Go&quot;">​</a></h3><ul><li><a href="https://github.com/sashabaranov/go-gpt3" target="_blank" rel="noopener noreferrer">go-gpt3</a> by <a href="https://github.com/sashabaranov" target="_blank" rel="noopener noreferrer">sashabaranov</a></li></ul><h3 id="java" tabindex="-1">Java <a class="header-anchor" href="#java" aria-label="Permalink to &quot;Java&quot;">​</a></h3><ul><li><a href="https://github.com/TheoKanning/openai-java" target="_blank" rel="noopener noreferrer">openai-java</a> by <a href="https://github.com/TheoKanning" target="_blank" rel="noopener noreferrer">Theo Kanning</a></li></ul><h3 id="julia" tabindex="-1">Julia <a class="header-anchor" href="#julia" aria-label="Permalink to &quot;Julia&quot;">​</a></h3><ul><li><a href="https://github.com/rory-linehan/OpenAI.jl" target="_blank" rel="noopener noreferrer">OpenAI.jl</a> by <a href="https://github.com/rory-linehan" target="_blank" rel="noopener noreferrer">rory-linehan</a></li></ul><h3 id="kotlin" tabindex="-1">Kotlin <a class="header-anchor" href="#kotlin" aria-label="Permalink to &quot;Kotlin&quot;">​</a></h3><ul><li><a href="https://github.com/Aallam/openai-kotlin" target="_blank" rel="noopener noreferrer">openai-kotlin</a> by <a href="https://github.com/Aallam" target="_blank" rel="noopener noreferrer">Mouaad Aallam</a></li></ul><h3 id="node-js" tabindex="-1">Node.js <a class="header-anchor" href="#node-js" aria-label="Permalink to &quot;Node.js&quot;">​</a></h3><ul><li><a href="https://www.npmjs.com/package/openai-api" target="_blank" rel="noopener noreferrer">openai-api</a> by <a href="https://github.com/Njerschow" target="_blank" rel="noopener noreferrer">Njerschow</a></li><li><a href="https://www.npmjs.com/package/openai-api-node" target="_blank" rel="noopener noreferrer">openai-api-node</a> by <a href="https://github.com/erlapso" target="_blank" rel="noopener noreferrer">erlapso</a></li><li><a href="https://www.npmjs.com/package/gpt-x" target="_blank" rel="noopener noreferrer">gpt-x</a> by <a href="https://github.com/ceifa" target="_blank" rel="noopener noreferrer">ceifa</a></li><li><a href="https://www.npmjs.com/package/gpt3" target="_blank" rel="noopener noreferrer">gpt3</a> by <a href="https://github.com/poteat" target="_blank" rel="noopener noreferrer">poteat</a></li><li><a href="https://www.npmjs.com/package/gpts" target="_blank" rel="noopener noreferrer">gpts</a> by <a href="https://github.com/thencc" target="_blank" rel="noopener noreferrer">thencc</a></li><li><a href="https://www.npmjs.com/package/@dalenguyen/openai" target="_blank" rel="noopener noreferrer">@dalenguyen/openai</a> by <a href="https://github.com/dalenguyen" target="_blank" rel="noopener noreferrer">dalenguyen</a></li><li><a href="https://github.com/tectalichq/public-openai-client-js" target="_blank" rel="noopener noreferrer">tectalic/openai</a> by <a href="https://tectalic.com/" target="_blank" rel="noopener noreferrer">tectalic</a></li></ul><h3 id="php" tabindex="-1">PHP <a class="header-anchor" href="#php" aria-label="Permalink to &quot;PHP&quot;">​</a></h3><ul><li><a href="https://packagist.org/packages/orhanerday/open-ai" target="_blank" rel="noopener noreferrer">orhanerday/open-ai</a> by <a href="https://github.com/orhanerday" target="_blank" rel="noopener noreferrer">orhanerday</a></li><li><a href="https://github.com/tectalichq/public-openai-client-php" target="_blank" rel="noopener noreferrer">tectalic/openai</a> by <a href="https://tectalic.com/" target="_blank" rel="noopener noreferrer">tectalic</a></li><li><a href="https://github.com/openai-php/client" target="_blank" rel="noopener noreferrer">openai-php clinet</a> by <a href="https://github.com/openai-php" target="_blank" rel="noopener noreferrer">openai-php</a></li></ul><h3 id="python" tabindex="-1">Python <a class="header-anchor" href="#python" aria-label="Permalink to &quot;Python&quot;">​</a></h3><ul><li><a href="https://github.com/OthersideAI/chronology" target="_blank" rel="noopener noreferrer">chronology</a> by <a href="https://www.othersideai.com/" target="_blank" rel="noopener noreferrer">OthersideAI</a></li></ul><h3 id="r" tabindex="-1">R <a class="header-anchor" href="#r" aria-label="Permalink to &quot;R&quot;">​</a></h3><ul><li><a href="https://github.com/ben-aaron188/rgpt3" target="_blank" rel="noopener noreferrer">rgpt3</a> by <a href="https://github.com/ben-aaron188" target="_blank" rel="noopener noreferrer">ben-aaron188</a></li></ul><h3 id="ruby" tabindex="-1">Ruby <a class="header-anchor" href="#ruby" aria-label="Permalink to &quot;Ruby&quot;">​</a></h3><ul><li><a href="https://github.com/nileshtrivedi/openai/" target="_blank" rel="noopener noreferrer">openai</a> by <a href="https://github.com/nileshtrivedi" target="_blank" rel="noopener noreferrer">nileshtrivedi</a></li><li><a href="https://github.com/alexrudall/ruby-openai" target="_blank" rel="noopener noreferrer">ruby-openai</a> by <a href="https://github.com/alexrudall" target="_blank" rel="noopener noreferrer">alexrudall</a></li></ul><h3 id="rust" tabindex="-1">Rust <a class="header-anchor" href="#rust" aria-label="Permalink to &quot;Rust&quot;">​</a></h3><ul><li><a href="https://github.com/64bit/async-openai" target="_blank" rel="noopener noreferrer">async-openai</a> by <a href="https://github.com/64bit" target="_blank" rel="noopener noreferrer">64bit</a></li><li><a href="https://github.com/lbkolev/fieri" target="_blank" rel="noopener noreferrer">fieri</a> by <a href="https://github.com/lbkolev" target="_blank" rel="noopener noreferrer">lbkolev</a></li></ul><h3 id="scala" tabindex="-1">Scala <a class="header-anchor" href="#scala" aria-label="Permalink to &quot;Scala&quot;">​</a></h3><ul><li><a href="https://github.com/cequence-io/openai-scala-client" target="_blank" rel="noopener noreferrer">openai-scala-client</a> by <a href="https://github.com/cequence-io" target="_blank" rel="noopener noreferrer">cequence-io</a></li></ul><h3 id="swift" tabindex="-1">Swift <a class="header-anchor" href="#swift" aria-label="Permalink to &quot;Swift&quot;">​</a></h3><ul><li><a href="https://github.com/dylanshine/openai-kit" target="_blank" rel="noopener noreferrer">OpenAIKit</a> by <a href="https://github.com/dylanshine" target="_blank" rel="noopener noreferrer">dylanshine</a></li></ul><h3 id="unity" tabindex="-1">Unity <a class="header-anchor" href="#unity" aria-label="Permalink to &quot;Unity&quot;">​</a></h3><ul><li><a href="https://github.com/hexthedev/OpenAi-Api-Unity" target="_blank" rel="noopener noreferrer">OpenAi-Api-Unity</a> by <a href="https://github.com/hexthedev" target="_blank" rel="noopener noreferrer">hexthedev</a></li><li><a href="https://github.com/RageAgainstThePixel/com.openai.unity" target="_blank" rel="noopener noreferrer">com.openai.unity</a> by <a href="https://github.com/RageAgainstThePixel" target="_blank" rel="noopener noreferrer">RageAgainstThePixel</a></li></ul><h3 id="unreal-engine" tabindex="-1">Unreal Engine <a class="header-anchor" href="#unreal-engine" aria-label="Permalink to &quot;Unreal Engine&quot;">​</a></h3><ul><li><a href="https://github.com/KellanM/OpenAI-Api-Unreal" target="_blank" rel="noopener noreferrer">OpenAI-Api-Unreal</a> by <a href="https://github.com/KellanM" target="_blank" rel="noopener noreferrer">KellanM</a></li></ul>`,58),o=[l];function t(p,i,c,h,b,y){return n(),e("div",null,o)}const g=a(s,[["render",t]]);export{D as __pageData,g as default};
