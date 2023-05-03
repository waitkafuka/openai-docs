import{_ as s,c as a,o as n,V as l}from"./chunks/framework.61392f5d.js";const u=JSON.parse('{"title":"内容分级","description":"","frontmatter":{},"headers":[],"relativePath":"guides/moderation.md","filePath":"guides/moderation.md","lastUpdated":null}'),p={name:"guides/moderation.md"},o=l(`<h1 id="内容分级" tabindex="-1">内容分级 <a class="header-anchor" href="#内容分级" aria-label="Permalink to &quot;内容分级&quot;">​</a></h1><h2 id="概述" tabindex="-1">概述 <a class="header-anchor" href="#概述" aria-label="Permalink to &quot;概述&quot;">​</a></h2><p>Moderation 端点是一个工具，您可以使用它来检查内容是否符合 OpenAI 的使用政策。开发人员可以识别违反我们使用政策的内容，并采取行动，例如过滤掉它。</p><p>该模型将分类以下类别：</p><table><thead><tr><th>类别</th><th>描述</th></tr></thead><tbody><tr><td>仇恨</td><td>表达、煽动或宣传基于种族、性别、族裔、宗教、国籍、性取向、残疾状况或种姓的仇恨的内容。</td></tr><tr><td>仇恨/威胁</td><td>包括针对目标群体的暴力或严重伤害的仇恨内容。</td></tr><tr><td>自残</td><td>促进、鼓励或描绘自残行为（如自杀、自残和饮食障碍）的内容。</td></tr><tr><td>性</td><td>旨在激起性欲的内容，如性行为描述或宣传性服务（不包括性教育和健康）。</td></tr><tr><td>性/未成年人</td><td>包括未满 18 岁的个人的性内容。</td></tr><tr><td>暴力</td><td>宣传或美化暴力或庆祝他人的痛苦或羞辱的内容。</td></tr><tr><td>暴力/图像</td><td>极其详细地描述死亡、暴力或严重身体伤害的暴力内容。</td></tr></tbody></table><p>在监控 OpenAI API 的输入和输出时，Moderation 端点是免费使用的。我们目前不支持监控第三方流量。</p><div class="info custom-block"><p class="custom-block-title">提示</p><p>我们不断努力提高分类器的准确性，特别是对仇恨、自残和暴力/图像内容的分类。我们对非英语语言的支持目前还比较有限。</p></div><h2 id="快速入门" tabindex="-1">快速入门 <a class="header-anchor" href="#快速入门" aria-label="Permalink to &quot;快速入门&quot;">​</a></h2><p>要获取文本的分级，可以像以下代码片段中所演示的那样向 <a href="https://platform.openai.com/docs/api-reference/moderations" target="_blank" rel="noreferrer">Moderation 端点</a>发出请求：</p><div class="vp-code-group"><div class="tabs"><input type="radio" name="group-T8cYg" id="tab-_DexDI8" checked="checked"><label for="tab-_DexDI8">bash</label><input type="radio" name="group-T8cYg" id="tab-uLA8SDV"><label for="tab-uLA8SDV">python</label></div><div class="blocks"><div class="language-bash active line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#FFCB6B;">curl</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">https://api.openai.com/v1/moderations</span><span style="color:#A6ACCD;"> \\</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#C3E88D;">-X</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">POST</span><span style="color:#A6ACCD;"> \\</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#C3E88D;">-H</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">Content-Type: application/json</span><span style="color:#89DDFF;">&quot;</span><span style="color:#A6ACCD;"> \\</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#C3E88D;">-H</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">Authorization: Bearer </span><span style="color:#A6ACCD;">$OPENAI_API_KEY</span><span style="color:#89DDFF;">&quot;</span><span style="color:#A6ACCD;"> \\</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#C3E88D;">-d</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">{&quot;input&quot;: &quot;Sample text goes here&quot;}</span><span style="color:#89DDFF;">&#39;</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br></div></div><div class="language-python line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#A6ACCD;">response </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> openai</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">Moderation</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">create</span><span style="color:#89DDFF;">(</span></span>
<span class="line"><span style="color:#82AAFF;">    </span><span style="color:#A6ACCD;font-style:italic;">input</span><span style="color:#89DDFF;">=</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">Sample text goes here</span><span style="color:#89DDFF;">&quot;</span></span>
<span class="line"><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">output </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> response</span><span style="color:#89DDFF;">[</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">results</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">][</span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">]</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br></div></div></div></div><p>以下是端点的示例输出。它返回以下字段：</p><ul><li><code>flagged</code>：如果模型将内容分类为违反 OpenAI 使用政策，则设置为 true；否则为 false。</li><li><code>categories</code>：包含每个类别的二进制使用政策违规标志的字典。对于每个类别，如果模型将相应类别标记为违规，则该值为 true，否则为 false。</li><li><code>category_scores</code>：包含模型输出的每个类别的原始得分字典，表示模型对输入是否违反 OpenAI 该类别的政策的自信程度。该值介于 0 和 1 之间，其中较高的值表示较高的置信度。这些得分不应被解释为概率。</li></ul><div class="language-json line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">json</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#89DDFF;">{</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C792EA;">id</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">modr-XXXXX</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C792EA;">model</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">text-moderation-001</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C792EA;">results</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">[</span></span>
<span class="line"><span style="color:#A6ACCD;">    </span><span style="color:#89DDFF;">{</span></span>
<span class="line"><span style="color:#A6ACCD;">      </span><span style="color:#89DDFF;">&quot;</span><span style="color:#FFCB6B;">categories</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">{</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">hate</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">hate/threatening</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">self-harm</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">sexual</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">sexual/minors</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">violence</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">violence/graphic</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false</span></span>
<span class="line"><span style="color:#A6ACCD;">      </span><span style="color:#89DDFF;">},</span></span>
<span class="line"><span style="color:#A6ACCD;">      </span><span style="color:#89DDFF;">&quot;</span><span style="color:#FFCB6B;">category_scores</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">{</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">hate</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.18805529177188873</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">hate/threatening</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.0001250059431185946</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">self-harm</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.0003706029092427343</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">sexual</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.0008735615410842001</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">sexual/minors</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.0007470346172340214</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">violence</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.0041268812492489815</span><span style="color:#89DDFF;">,</span></span>
<span class="line"><span style="color:#A6ACCD;">        </span><span style="color:#89DDFF;">&quot;</span><span style="color:#F78C6C;">violence/graphic</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">0.00023186142789199948</span></span>
<span class="line"><span style="color:#A6ACCD;">      </span><span style="color:#89DDFF;">},</span></span>
<span class="line"><span style="color:#A6ACCD;">      </span><span style="color:#89DDFF;">&quot;</span><span style="color:#FFCB6B;">flagged</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">:</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">false</span></span>
<span class="line"><span style="color:#A6ACCD;">    </span><span style="color:#89DDFF;">}</span></span>
<span class="line"><span style="color:#A6ACCD;">  </span><span style="color:#89DDFF;">]</span></span>
<span class="line"><span style="color:#89DDFF;">}</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br><span class="line-number">6</span><br><span class="line-number">7</span><br><span class="line-number">8</span><br><span class="line-number">9</span><br><span class="line-number">10</span><br><span class="line-number">11</span><br><span class="line-number">12</span><br><span class="line-number">13</span><br><span class="line-number">14</span><br><span class="line-number">15</span><br><span class="line-number">16</span><br><span class="line-number">17</span><br><span class="line-number">18</span><br><span class="line-number">19</span><br><span class="line-number">20</span><br><span class="line-number">21</span><br><span class="line-number">22</span><br><span class="line-number">23</span><br><span class="line-number">24</span><br><span class="line-number">25</span><br><span class="line-number">26</span><br><span class="line-number">27</span><br></div></div><div class="info custom-block"><p class="custom-block-title">提示</p><p>OpenAI 将不断升级 Moderation 端点的底层模型。因此，依赖 <code>category_scores</code> 的自定义政策可能需要随时间重新校准。</p></div>`,14),e=[o];function t(r,c,D,F,y,i){return n(),a("div",null,e)}const A=s(p,[["render",t]]);export{u as __pageData,A as default};
