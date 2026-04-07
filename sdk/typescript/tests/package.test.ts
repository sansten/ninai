import pkg from "../package.json";
import {
  NinaiClient,
  CognitiveGateway,
  MemoryResource,
} from "../src/index.js";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function isSemver(value: string): boolean {
  return /^\d+\.\d+\.\d+$/.test(value);
}

assert(pkg.name === "@ninai/sdk", "package name must be @ninai/sdk");
assert(isSemver(pkg.version), "version must be semver x.y.z");
assert(pkg.main === "dist/index.js", "main must point to dist/index.js");
assert(pkg.types === "dist/index.d.ts", "types must point to dist/index.d.ts");
assert(typeof NinaiClient === "function", "NinaiClient must be exported");
assert(typeof CognitiveGateway === "function", "CognitiveGateway must be exported");
assert(typeof MemoryResource === "function", "MemoryResource must be exported");
assert(pkg.engines?.node === ">=18", "engines.node must be >=18");
