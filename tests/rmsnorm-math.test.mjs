import test from 'node:test';
import assert from 'node:assert/strict';
import { computeRmsNorm, rms } from '../src/rmsnorm-math.ts';

const near = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test('[3,4] reproduces the worked example and preserves direction', () => {
  const result = computeRmsNorm([3, 4], [1, 1], 1, 1e-6);
  near(result.meanSquare, 12.5);
  near(result.output[0], 3 / Math.sqrt(12.500001));
  near(result.output[1], 4 / Math.sqrt(12.500001));
  near(result.output[0] / result.output[1], 3 / 4);
  near(rms(result.normalized), Math.sqrt(12.5 / 12.500001));
});

test('positive scaling changes the input norm but approximately preserves normalized output', () => {
  const small = computeRmsNorm([3, 4], [1, 1], 1, 1e-6);
  const large = computeRmsNorm([3, 4], [1, 1], 10, 1e-6);
  near(rms(large.input), rms(small.input) * 10);
  small.output.forEach((value, i) => near(value, large.output[i], 1e-6));
});

test('zero input stays zero and epsilon keeps the reciprocal finite', () => {
  const result = computeRmsNorm([0, 0], [2, 3], 1, 1e-6);
  assert.deepEqual(result.output, [0, 0]);
  near(result.reciprocal, 1000);
});

test('per-feature gamma can change output direction and RMS', () => {
  const result = computeRmsNorm([3, -4, 2, -1], [2, -1, 0, 1], 1, 1e-6);
  near(result.output[0], result.normalized[0] * 2);
  near(result.output[1], -result.normalized[1]);
  near(result.output[2], 0);
  assert.ok(Math.abs(rms(result.output) - 1) > 0.1);
});

test('epsilon-dominated inputs are not incorrectly reported as unit RMS', () => {
  const result = computeRmsNorm([1e-6, 1e-6], [1, 1], 1, 1);
  assert.ok(rms(result.normalized) < 0.00001);
});

test('invalid dimensions and non-finite values are rejected', () => {
  for (const args of [[[], [], 1, 1e-6], [[1], [], 1, 1e-6], [[NaN], [1], 1, 1e-6], [[1], [1], 1, 0]]) {
    assert.throws(() => computeRmsNorm(...args));
  }
});
