---
---

# VitePress 💙 StackBlitz

Hi there :wave: This is a demo running VitePress within your **browser tab**!

::: code-group

```sh [npm]
$ npm install -D vitepress
```

```sh [pnpm]
$ pnpm add -D vitepress
```

```sh [yarn]
$ yarn add -D vitepress
```

:::

::: details Getting missing peer deps warnings?
If using PNPM, you will notice a missing peer warning for `@docsearch/js`. This does not prevent VitePress from working. If you wish to suppress this warning, add the following to your `package.json`:

```json
"pnpm": {
  "peerDependencyRules": {
    "ignoreMissing": [
      "@algolia/client-search"
    ]
  }
}
```

:::

## Powered by Vite

VitePress uses Vite under the hood. This means:

:::tip
By default, VitePress stores its dev server cache in `.vitepress/cache`, and the production build output in `.vitepress/dist`. If using Git, you should add them to your `.gitignore` file. These locations can also be [configured](/reference/site-config#outdir).
:::

- Instant server start
- Lightning fast HMR
- Optimized builds

## Markdown-Centered

**Input**

```md
::: info
This is an info box.
:::

::: tip
This is a tip.
:::

::: warning
This is a warning.
:::

::: danger
This is a dangerous warning.
:::

::: details
This is a details block.
:::
```

**Output**

::: info
This is an info box.
:::

::: tip
This is a tip.
:::

::: warning
This is a warning.
:::

::: danger
This is a dangerous warning.
:::

::: details
This is a details block.
:::

So you can focus more on writing. Powered by MarkdownIt. Comes with many [built-in extensions](https://vitepress.vuejs.org/guide/markdown), and you can use Vue features in Markdown too!

:::tip SSR Compatibility
All Vue usage needs to be SSR-compatible. See <a href="./ssr-compat" target="_blank" rel="noreferrer">./ssr-compat</a> for details and common workarounds.
:::

<div class="escape-demo">

::: v-pre
{{ This will be displayed as-is }}
:::

</div>

### Title <Badge type="info" text="default" />

### Title <Badge type="tip" text="^1.9.0" />

### Title <Badge type="warning" text="beta" />

### Title <Badge type="danger" text="caution" />
