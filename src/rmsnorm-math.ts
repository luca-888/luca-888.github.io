export function rms(values: readonly number[]) {
  return Math.hypot(...values) / Math.sqrt(values.length);
}

export function computeRmsNorm(base: readonly number[], weights: readonly number[], scale: number, epsilon: number) {
  if (!base.length || base.length !== weights.length || ![...base, ...weights, scale, epsilon].every(Number.isFinite) || epsilon <= 0) {
    throw new Error('RMSNorm requires finite, equal-length vectors and a positive epsilon.');
  }
  const input = base.map((value) => value * scale);
  const meanSquare = input.reduce((sum, value) => sum + value * value, 0) / input.length;
  const denominator = Math.sqrt(meanSquare + epsilon);
  const reciprocal = 1 / denominator;
  const normalized = input.map((value) => value * reciprocal);
  const output = normalized.map((value, index) => value * weights[index]);
  return { input, meanSquare, denominator, reciprocal, normalized, output };
}

export function formatValue(value: number) {
  if (Object.is(value, -0) || value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 100000) return value.toExponential(3);
  return Number(value.toFixed(4)).toString();
}
