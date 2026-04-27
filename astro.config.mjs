import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://luca-888.github.io",
  markdown: {
    shikiConfig: {
      theme: "github-light",
      wrap: true,
    },
  },
});
