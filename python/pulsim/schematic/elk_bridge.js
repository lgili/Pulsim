// elk_bridge.js — JS subprocess bridge for pulsim.schematic's ELK backend.
//
// Reads an ELK graph JSON document from stdin, runs the layered Sugiyama
// layout (with orthogonal edge routing), and writes the laid-out graph to
// stdout. Errors go to stderr with a non-zero exit code.
//
// The Python side (elk_backend.py) is responsible for building the input
// JSON and parsing the output. This script is intentionally a thin
// passthrough so the layout engine version can be pinned in package.json
// without coupling Python releases to ELK upgrades.
//
// Usage (from Python):
//     node elk_bridge.js < input.json > output.json

"use strict";

const ELK = require("elkjs");

async function main() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const inputText = Buffer.concat(chunks).toString("utf8");
  if (inputText.trim().length === 0) {
    throw new Error("elk_bridge: empty input on stdin");
  }
  const graph = JSON.parse(inputText);
  const elk = new ELK();
  const result = await elk.layout(graph);
  process.stdout.write(JSON.stringify(result));
}

main().catch((err) => {
  process.stderr.write(`elk_bridge error: ${err && err.message ? err.message : err}\n`);
  process.exit(1);
});
