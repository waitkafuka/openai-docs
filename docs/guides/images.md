# 图片生成 <Badge text="Beta" type="tip"/>

学习如何使用我们的 DALL·E 模型生成或操作图像

## 介绍

图片生成 API 提供了与图像交互的三种方法：

- 根据文本提示从头开始创建图像
- 基于文本提示编辑现有图像
- 创建现有图像的变体

本指南涵盖了使用这三个 API 端点的基础知识，并提供有用的代码示例。要查看它们的实际效果，请查看我们的[ DALL·E 预览应用程序](https://labs.openai.com/)。

:::warning 速率限制
图片 API 处于 beta 版本。在此期间，API 和模型将根据您的反馈进行演变。为确保所有用户可以舒适地原型制作，默认速率限制为每分钟 50 张图像。您可以在我们的速率限制指南中了解更多有关速率限制的信息。
:::

## 使用方法

### 生成

图像生成端点允许您根据文本提示创建原始图像。生成的图像可以具有 256x256、512x512 或 1024x1024 像素的大小。较小的尺寸生成更快。您可以使用 n 参数一次请求 1 到 10 张图像。

:::code-group

```python [python]
response = openai.Image.create(
  prompt="a white siamese cat",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
```

```javascript [nodejs]
const response = await openai.createImage({
  prompt: 'a white siamese cat',
  n: 1,
  size: '1024x1024',
});
image_url = response.data.data[0].url;
```

```sh [curl]
curl https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "a white siamese cat",
    "n": 1,
    "size": "1024x1024"
  }'
```

:::

描述越详细，越可能得到你想要的结果。你可以在 DALL·E 预览应用程序中探索更多的示例，以获得更多的启发。以下是一个快速的例子：

<div class="images-examples flex-first-col"><table><thead><tr><th>Prompt</th><th>结果</th></tr></thead><tbody><tr><td>a white siamese cat</td><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_generation_simple.webp"></td></tr><tr><td>a close up, studio photographic portrait of a white siamese cat that looks curious, backlit ears</td><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_generation_detailed.webp"></td></tr></tbody></table></div>

每个图像都可以使用 [`response_format`](https://platform.openai.com/docs/api-reference/images/create#images/create-response_format)参数指定返回类型为`URL`或者`base64`。`URL` 将在一小时后失效。

### 编辑图像

图像编辑端点允许您通过上传蒙版来编辑和扩展图像。蒙版的透明区域表示应该编辑图像的位置，提示应该描述完整的新图像，而不仅仅是擦除的区域。此端点可以实现类似于我们的 DALL·E 预览应用程序中的编辑器的体验。

:::code-group

```python [python]
response = openai.Image.create_edit(
  image=open("sunlit_lounge.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="A sunlit indoor lounge area with a pool containing a flamingo",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
```

```javascript [nodejs]
const response = await openai.createImageEdit(fs.createReadStream('sunlit_lounge.png'), fs.createReadStream('mask.png'), 'A sunlit indoor lounge area with a pool containing a flamingo', 1, '1024x1024');
image_url = response.data.data[0].url;
```

```sh [curl]
    curl https://api.openai.com/v1/images/edits \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image="@sunlit_lounge.png" \
  -F mask="@mask.png" \
  -F prompt="A sunlit indoor lounge area with a pool containing a flamingo" \
  -F n=1 \
  -F size="1024x1024"
```

:::

<div class="images-examples"><table><thead><tr><th>Image</th><th>Mask</th><th>Output</th></tr></thead><tbody><tr><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_edit_original.webp"></td><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_edit_mask.webp"></td><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_edit_output.webp"></td></tr></tbody></table></div>
<p class="images-edit-prompt body-small" style="color:#999">Prompt: a sunlit indoor lounge area with a pool containing a flamingo</p>

上传的图像和遮罩必须都是小于 4MB 的正方形 PNG 图像，并且它们的尺寸必须相同。当生成输出时，遮罩的非透明区域不会被使用，因此非透明区域不需要同原始图像一致。

### 变体

图像变化端点允许您生成给定图像的变化。

:::code-group

```python [python]
response = openai.Image.create_variation(
  image=open("corgi_and_cat_paw.png", "rb"),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
```

```javascript [nodejs]
const response = await openai.createImageVariation(fs.createReadStream('corgi_and_cat_paw.png'), 1, '1024x1024');
image_url = response.data.data[0].url;
```

```sh [curl]
curl https://api.openai.com/v1/images/variations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image='@corgi_and_cat_paw.png' \
  -F n=1 \
  -F size="1024x1024"
```

:::

<div class="images-examples"><table><thead><tr><th>Image</th><th>Output</th></tr></thead><tbody><tr><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_variation_original.webp"></td><td><img class="images-example-image" src="https://cdn.openai.com/API/images/guides/image_variation_output.webp"></td></tr></tbody></table></div>

<p style>与编辑端点类似，输入图像必须是小于4MB的正方形PNG图像。</p>

### 内容政策

基于我们的内容政策，提示和图片将根据过滤器进行筛选并返回错误。如果您对误报或其他相关问题有任何反馈，请通过我们的帮助中心与我们联系。

## 编程语言指南

### nodejs 版本

#### 使用内存中的图像数据

上面指南中的 Node.js 示例使用 fs 模块从磁盘读取图像数据。在某些情况下，您可能的图像数据位于内存中。以下是一个使用存储在 Node.js 缓冲对象中的图像数据的示例 API 调用：

```javascript [nodejs]
// This is the Buffer object that contains your image data
const buffer = [your image data];
// Set a `name` that ends with .png so that the API knows it's a PNG image
buffer.name = "image.png";
const response = await openai.createImageVariation(
  buffer,
  1,
  "1024x1024"
);
```

#### 使用 TypeScript

如果您正在使用 TypeScript，则可能会遇到图像文件参数的一些奇怪问题。以下是通过显式转换参数来解决类型不匹配的示例：

```typescript [typescript]
// Cast the ReadStream to `any` to appease the TypeScript compiler
const response = await openai.createImageVariation(fs.createReadStream('image.png') as any, 1, '1024x1024');
```

以下是用于内存中图像数据的一个类似的例子：

```typescript [typescript]
// This is the Buffer object that contains your image data
const buffer: Buffer = [your image data];
// Cast the buffer to `any` so that we can set the `name` property
const file: any = buffer;
// Set a `name` that ends with .png so that the API knows it's a PNG image
file.name = "image.png";
const response = await openai.createImageVariation(
  file,
  1,
  "1024x1024"
);
```

#### 错误处理

API 请求可能因无效输入、速率限制或其他问题而返回错误。这些错误可以使用 try...catch 语句处理，并且错误细节可以在 error.response 或 error.message 中找到：

```javascript [nodejs]
try {
  const response = await openai.createImageVariation(fs.createReadStream('image.png'), 1, '1024x1024');
  console.log(response.data.data[0].url);
} catch (error) {
  if (error.response) {
    console.log(error.response.status);
    console.log(error.response.data);
  } else {
    console.log(error.message);
  }
}
```

### Python

#### 使用内存中的图像数据

```python [python]
from io import BytesIO

# This is the BytesIO object that contains your image data
byte_stream: BytesIO = [your image data]
byte_array = byte_stream.getvalue()
response = openai.Image.create_variation(
  image=byte_array,
  n=1,
  size="1024x1024"
)
```

#### 操作内存中的图像

在传给 API 之前，对图像进行优化是有用的。下面是一个使用 `PIL` 来调整图像大小的例子：

```python [python]
from io import BytesIO
from PIL import Image

# 从磁盘上读取图像文件并调整大小

image = Image.open("image.png")
width, height = 256, 256
image = image.resize((width, height))

# 将图像转换为 BytesIO 对象

byte_stream = BytesIO()
image.save(byte_stream, format='PNG')
byte_array = byte_stream.getvalue()

response = openai.Image.create_variation(
image=byte_array,
n=1,
size="1024x1024"
)
```

#### 错误处理

API 请求可能会因为无效输入、速率限制或其他问题而返回错误。这些错误可以使用 try...except 语句处理，错误详细信息可以在 e.error 中找到：

```python [python]
try:
openai.Image.create_variation(
open("image.png", "rb"),
n=1,
size="1024x1024"
)
print(response['data'][0]['url'])
except openai.error.OpenAIError as e:
print(e.http_status)
print(e.error)
```
