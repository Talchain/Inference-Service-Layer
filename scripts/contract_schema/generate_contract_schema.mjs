#!/usr/bin/env node
/**
 * generate_contract_schema.mjs — derive the @talchain/schemas contract as JSON Schema.
 *
 * DERIVE, DON'T MIRROR: this script is the ONLY way the committed artifact at
 * tests/fixtures/contract-schema/talchain-schemas.json may be produced. It runs
 * inside a BUILT checkout of Talchain/olumi-schemas at the ref pinned in
 * tests/fixtures/contract-schema/PIN.json, walks every Zod export of the package
 * root and boundary entry points, and converts each to JSON Schema with a PINNED
 * zod-to-json-schema version. Output is byte-deterministic (sorted keys, no
 * timestamps) so the CI freshness gate can compare bytes.
 *
 * Usage (from a built olumi-schemas checkout — `npm ci && npm run build` and
 * `npm install --no-save zod-to-json-schema@<pinned>` must have run there):
 *   node <isl>/scripts/contract_schema/generate_contract_schema.mjs \
 *     --schemas-dir /path/to/olumi-schemas \
 *     --ref <sha-or-tag> \
 *     --out <isl>/tests/fixtures/contract-schema/talchain-schemas.json
 *
 * Or use scripts/contract_schema/refresh_contract_schema.sh which does all of it.
 */

import { createRequire } from 'node:module';
import { writeFileSync, readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= process.argv.length) {
    console.error(`Missing required argument --${name}`);
    process.exit(2);
  }
  return process.argv[i + 1];
}

const schemasDir = path.resolve(arg('schemas-dir'));
const ref = arg('ref');
const outPath = path.resolve(arg('out'));

// Resolve zod-to-json-schema FROM THE SCHEMAS CHECKOUT so it shares the zod
// instance the schemas were built against.
const requireFromSchemas = createRequire(path.join(schemasDir, 'package.json'));
const z2jsEntry = requireFromSchemas.resolve('zod-to-json-schema');
// package.json is not an exported subpath — read it directly from the package dir.
const z2jsPkgDir = z2jsEntry.slice(0, z2jsEntry.lastIndexOf('node_modules/zod-to-json-schema')) +
  'node_modules/zod-to-json-schema';
const z2jsPkg = JSON.parse(readFileSync(path.join(z2jsPkgDir, 'package.json'), 'utf8'));
const { zodToJsonSchema } = await import(pathToFileURL(z2jsEntry).href);

const pkg = JSON.parse(readFileSync(path.join(schemasDir, 'package.json'), 'utf8'));

const entryPoints = {
  index: path.join(schemasDir, 'dist', 'index.js'),
  boundary: path.join(schemasDir, 'dist', 'boundary', 'index.js'),
};

function isZodSchema(v) {
  return (
    v !== null &&
    typeof v === 'object' &&
    '_def' in v &&
    typeof v.safeParse === 'function'
  );
}

/** Recursively sort object keys for byte-deterministic output. Arrays keep order. */
function sortDeep(value) {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value !== null && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = sortDeep(value[k]);
    return out;
  }
  return value;
}

const modules = {};
for (const [moduleName, entry] of Object.entries(entryPoints)) {
  const ns = await import(pathToFileURL(entry).href);
  const schemas = {};
  for (const exportName of Object.keys(ns).sort()) {
    const v = ns[exportName];
    if (!isZodSchema(v)) continue;
    let js;
    try {
      // $refStrategy 'none' inlines nested schemas — every property carries its
      // full type locally, which is what the drift differ needs.
      js = zodToJsonSchema(v, { $refStrategy: 'none' });
    } catch (err) {
      js = { 'x-conversion-error': String(err) };
    }
    delete js.$schema; // identical on every entry; noise for the differ
    schemas[exportName] = js;
  }
  modules[moduleName] = schemas;
}

const artifact = sortDeep({
  _meta: {
    source_repo: 'Talchain/olumi-schemas',
    source_ref: ref,
    package_name: pkg.name,
    package_version: pkg.version,
    generator: 'Inference-Service-Layer/scripts/contract_schema/generate_contract_schema.mjs',
    'zod-to-json-schema': z2jsPkg.version,
    determinism: 'sorted-keys, no timestamps — byte-comparable by the CI freshness gate',
  },
  modules,
});

writeFileSync(outPath, JSON.stringify(artifact, null, 2) + '\n');
const count = Object.values(modules).reduce((a, m) => a + Object.keys(m).length, 0);
console.log(`Wrote ${count} schemas from ${pkg.name}@${pkg.version} (ref ${ref}) to ${outPath}`);
