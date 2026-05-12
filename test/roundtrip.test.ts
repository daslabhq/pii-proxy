import { describe, test, expect } from 'bun:test';
import { PrivacyProxy } from '../src/index.js';

describe('PrivacyProxy', () => {
  test('mask and unmask email round-trip', () => {
    const proxy = new PrivacyProxy();
    const original = 'Contact alex@example.com for details';

    const masked = proxy.mask(original);
    expect(masked.text).not.toContain('alex@example.com');
    expect(masked.detections).toHaveLength(1);
    expect(masked.detections[0].type).toBe('email');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('same email always maps to same fake', () => {
    const proxy = new PrivacyProxy();

    const r1 = proxy.mask('email alex@example.com');
    const r2 = proxy.mask('also alex@example.com here');

    const fake1 = r1.detections[0].replacement;
    const fake2 = r2.detections[0].replacement;
    expect(fake1).toBe(fake2);
  });

  test('different emails map to different fakes', () => {
    const proxy = new PrivacyProxy();

    const r1 = proxy.mask('alex@example.com');
    const r2 = proxy.mask('jordan@example.com');

    expect(r1.detections[0].replacement).not.toBe(r2.detections[0].replacement);
  });

  test('mask and unmask tracking number round-trip', () => {
    const proxy = new PrivacyProxy();
    const original = 'Your package AETH0000345323DY has shipped';

    const masked = proxy.mask(original);
    expect(masked.text).not.toContain('AETH0000345323DY');
    expect(masked.detections).toHaveLength(1);
    expect(masked.detections[0].type).toBe('tracking_number');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('mask and unmask multiple entity types', () => {
    const proxy = new PrivacyProxy();
    const original = 'Email alex@example.com, tracking AETH0000345323DY, IP 192.168.1.1';

    const masked = proxy.mask(original);
    expect(masked.text).not.toContain('alex@example.com');
    expect(masked.text).not.toContain('AETH0000345323DY');
    expect(masked.text).not.toContain('192.168.1.1');
    expect(masked.detections).toHaveLength(3);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('mask preserves non-PII text exactly', () => {
    const proxy = new PrivacyProxy();
    const original = 'Hello world, this has no PII at all!';

    const masked = proxy.mask(original);
    expect(masked.text).toBe(original);
    expect(masked.detections).toHaveLength(0);
  });

  test('unmask handles LLM response with masked values', () => {
    const proxy = new PrivacyProxy();

    // Step 1: mask the input
    const input = proxy.mask('Ship order to alex@example.com');
    const fakeEmail = input.detections[0].replacement;

    // Step 2: simulate LLM response that references the fake email
    const llmResponse = `I've sent a shipping notification to ${fakeEmail}. They should receive it shortly.`;

    // Step 3: unmask the response
    const restored = proxy.unmask(llmResponse);
    expect(restored).toContain('alex@example.com');
    expect(restored).not.toContain(fakeEmail);
  });

  test('maskObject handles nested structures', () => {
    const proxy = new PrivacyProxy();

    const input = {
      to: 'alex@example.com',
      subject: 'Order update',
      body: 'Tracking: AETH0000345323DY',
      metadata: {
        ip: '10.0.0.1',
      },
    };

    const { masked, detections } = proxy.maskObject(input);
    expect(masked.to).not.toContain('alex@example.com');
    expect(masked.body).not.toContain('AETH0000345323DY');
    expect(masked.metadata.ip).not.toBe('10.0.0.1');
    expect(masked.subject).toBe('Order update'); // no PII here

    const restored = proxy.unmaskObject(masked);
    expect(restored).toEqual(input);
  });

  test('map serialization and restoration', () => {
    const proxy1 = new PrivacyProxy();
    const masked = proxy1.mask('alex@example.com');
    const serialized = proxy1.getMap().serialize();

    // New proxy with restored map
    const proxy2 = new PrivacyProxy();
    proxy2.loadMap(serialized);
    const restored = proxy2.unmask(masked.text);
    expect(restored).toBe('alex@example.com');
  });

  test('UUID detection and round-trip', () => {
    const proxy = new PrivacyProxy();
    const original = 'User ID: 550e8400-e29b-41d4-a716-446655440000';

    const masked = proxy.mask(original);
    expect(masked.text).not.toContain('550e8400-e29b-41d4-a716-446655440000');
    expect(masked.detections[0].type).toBe('uuid');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('credit card detection with Luhn validation', () => {
    const proxy = new PrivacyProxy();
    // Valid Luhn number
    const original = 'Card: 4532015112830366';

    const masked = proxy.mask(original);
    expect(masked.detections.length).toBeGreaterThanOrEqual(1);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('URL with tokens', () => {
    const proxy = new PrivacyProxy();
    const original = 'Click https://api.example.com/verify?token=abc123secret&user=456';

    const masked = proxy.mask(original);
    expect(masked.text).not.toContain('abc123secret');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('multiple occurrences of same entity', () => {
    const proxy = new PrivacyProxy();
    const original = 'From alex@example.com to alex@example.com';

    const masked = proxy.mask(original);
    const parts = masked.text.split(' to ');
    // Both occurrences should map to the same fake
    expect(parts[0].replace('From ', '')).toBe(parts[1]);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });
});
