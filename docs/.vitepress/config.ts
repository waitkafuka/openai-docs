import { defineConfig } from 'vitepress';
// refer https://vitepress.vuejs.org/config/introduction for details
export default defineConfig({

  ignoreDeadLinks: true,
  lang: 'zh-cn',
  title: 'OpenAI API中文文档',
  appearance: false,
  lastUpdated: true,
  description: 'ChatGPT, OpenAI API中文文档',
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'Keywords', content: 'ChatGPT,OpenAI,OpenAI API,OpenAI API中文网' }],
    ['link', { rel: 'stylesheet', href: '/styles/openai-main.css' }],
    ['link', { rel: 'stylesheet', href: '/styles/openai.css' }],
    ['link', { rel: 'stylesheet', href: '/styles/global.css' }],
  ],
  markdown: {
    lineNumbers: true
  },
  themeConfig: {
    search: {
      provider: 'local'
    },
    editLink: {
      pattern: 'https://github.com/vuejs/vitepress/edit/main/docs/:path',
      text: '在GitHub上编辑',
    },

    darkModeSwitchLabel: '切换暗黑模式',
    outline: 'deep',
    // aside: false,
    nav: nav(),

    sidebar: [
      {
        text: '起步',
        items: [
          { text: '概述', link: '/introduction' },
          { text: '快速开始', link: '/quickstart' },
          { text: '资源库', link: '/libraries' },
          { text: '模型', link: '/models' },
          { text: '实战演示', link: '/tutorials' },
          // { text: 'example', link: '/example' },
          // ...
        ],
      },
      {
        text: '引导',
        items: [
          { text: '文本补全', link: '/guides/completion' },
          { text: '聊天功能', link: '/guides/chat' },
          { text: '图片生成', link: '/guides/images' },
          { text: '微调（Fine-tuning）', link: '/guides/fine-tuning' },
          { text: '向量化（Embeddings）', link: '/guides/embeddings' },
          { text: '语音转文字', link: '/guides/speech-to-text' },
          { text: '内容分级', link: '/guides/moderation' },
          { text: '限流(待翻译)', link: '/building' },
          { text: '错误码(待翻译)', link: '/building' },
          { text: '安全最佳实践(待翻译)', link: '/building' },
          { text: '生产环境最佳实践(待翻译)', link: '/building' },
          // ...
        ],
      },
      // {
      //   text: '聊天插件',
      //   items: [
      //     { text: '介绍', link: '/introduction' },
      //     { text: '起步', link: '/quickstart' },
      //     { text: '身份认证', link: '/libraries' },
      //     { text: '示例', link: '/models' },
      //     { text: '生产环境', link: '/tutorials' },
      //     { text: '插件预览', link: '/example' },
      //     { text: '插件政策', link: '/example' },
      //     // ...
      //   ],
      // },
    ],
  },
});

function nav() {
  return [
    { text: '概要说明', link: '/introduction' },
    { text: 'API文档', link: '/api-reference' },

    // {
    //   text: 'Dropdown Menu',
    //   items: [
    //     { text: 'Item A', link: '/item-1' },
    //     { text: 'Item B', link: '/item-2' },
    //     { text: 'Item C', link: '/item-3' },
    //   ],
    // },

    // ...
  ];
}
