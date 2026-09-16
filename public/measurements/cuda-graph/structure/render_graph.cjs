// npm install @viz-js/viz@3.30.0 in a separate tooling directory.
const fs = require('node:fs');
const path = require('node:path');
const { instance } = require(process.argv[2] || '@viz-js/viz');

instance().then(viz => {
  const dot = fs.readFileSync(path.join(__dirname, 'graph.dot'), 'utf8');
  const svg = viz.renderString(dot, { format: 'svg', engine: 'dot' });
  fs.writeFileSync(path.join(__dirname, 'graph.svg'), svg);
  console.log(`Rendered original DOT with Graphviz ${viz.graphvizVersion}.`);
});
