import { describe, test, expect } from 'bun:test';
import { PrivacyProxy } from '../src/index.js';

describe('PrivacyProxy', () => {
  test('mask and unmask email round-trip', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Contact alex@example.com for details';

    const masked = await proxy.mask(original);
    expect(masked.text).not.toContain('alex@example.com');
    expect(masked.detections).toHaveLength(1);
    expect(masked.detections[0].type).toBe('email');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('same email always maps to same fake', async () => {
    const proxy = new PrivacyProxy();

    const r1 = await proxy.mask('email alex@example.com');
    const r2 = await proxy.mask('also alex@example.com here');

    const fake1 = r1.detections[0].replacement;
    const fake2 = r2.detections[0].replacement;
    expect(fake1).toBe(fake2);
  });

  test('different emails map to different fakes', async () => {
    const proxy = new PrivacyProxy();

    const r1 = await proxy.mask('alex@example.com');
    const r2 = await proxy.mask('jordan@example.com');

    expect(r1.detections[0].replacement).not.toBe(r2.detections[0].replacement);
  });

  test('mask and unmask tracking number round-trip', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Your package AETH0000345323DY has shipped';

    const masked = await proxy.mask(original);
    expect(masked.text).not.toContain('AETH0000345323DY');
    expect(masked.detections).toHaveLength(1);
    expect(masked.detections[0].type).toBe('tracking_number');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('mask and unmask IPv6 round-trip', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Login from 2001:db8:85a3::8a2e:370:7334 and fe80::1';

    const masked = await proxy.mask(original);
    expect(masked.text).not.toContain('2001:db8:85a3::8a2e:370:7334');
    expect(masked.text).not.toContain('fe80::1');
    expect(masked.detections).toHaveLength(2);
    expect(masked.detections.every(d => d.type === 'ip_address')).toBe(true);
    // fakes keep the address family
    expect(masked.detections.every(d => d.replacement.includes(':'))).toBe(true);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('IPv6 detector ignores times and ratios', async () => {
    const proxy = new PrivacyProxy();
    const masked = await proxy.mask('Meeting at 12:30, score was 3:2, uptime 99:99:99');
    expect(masked.detections).toHaveLength(0);
    expect(masked.text).toBe('Meeting at 12:30, score was 3:2, uptime 99:99:99');
  });

  test('context-gated ID detection round-trip', async () => {
    const proxy = new PrivacyProxy();
    const original =
      'Passport number HA3552738, SSN 539-75-3166, driver license WDLABCD456DG, ID card AB12345CD';

    const masked = await proxy.mask(original);
    const types = masked.detections.map(d => d.type).sort();
    expect(types).toEqual(['driver_license', 'id_card', 'national_id', 'passport_number']);
    expect(masked.text).not.toContain('HA3552738');
    expect(masked.text).not.toContain('539-75-3166');
    // format-preserving fakes keep the shape
    const passport = masked.detections.find(d => d.type === 'passport_number')!;
    expect(passport.replacement).toMatch(/^[A-Z]{2}\d{7}$/);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('ID detector needs context — bare numbers are not IDs', async () => {
    const proxy = new PrivacyProxy();
    const masked = await proxy.mask('Order total was 539751234, room 4521, year 2024');
    const idTypes = masked.detections.filter(d =>
      ['passport_number', 'national_id', 'driver_license', 'id_card'].includes(d.type));
    expect(idTypes).toHaveLength(0);
  });

  test('mask and unmask multiple entity types', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Email alex@example.com, tracking AETH0000345323DY, IP 192.168.1.1';

    const masked = await proxy.mask(original);
    expect(masked.text).not.toContain('alex@example.com');
    expect(masked.text).not.toContain('AETH0000345323DY');
    expect(masked.text).not.toContain('192.168.1.1');
    expect(masked.detections).toHaveLength(3);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('mask preserves non-PII text exactly', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Hello world, this has no PII at all!';

    const masked = await proxy.mask(original);
    expect(masked.text).toBe(original);
    expect(masked.detections).toHaveLength(0);
  });

  test('unmask handles LLM response with masked values', async () => {
    const proxy = new PrivacyProxy();

    const input = await proxy.mask('Ship order to alex@example.com');
    const fakeEmail = input.detections[0].replacement;

    const llmResponse = `I've sent a shipping notification to ${fakeEmail}. They should receive it shortly.`;

    const restored = proxy.unmask(llmResponse);
    expect(restored).toContain('alex@example.com');
    expect(restored).not.toContain(fakeEmail);
  });

  test('maskObject handles nested structures', async () => {
    const proxy = new PrivacyProxy();

    const input = {
      to: 'alex@example.com',
      subject: 'Order update',
      body: 'Tracking: AETH0000345323DY',
      metadata: {
        ip: '10.0.0.1',
      },
    };

    const { masked, detections } = await proxy.maskObject(input);
    expect(masked.to).not.toContain('alex@example.com');
    expect(masked.body).not.toContain('AETH0000345323DY');
    expect(masked.metadata.ip).not.toBe('10.0.0.1');
    expect(masked.subject).toBe('Order update');

    const restored = proxy.unmaskObject(masked);
    expect(restored).toEqual(input);
  });

  test('map serialization and restoration', async () => {
    const proxy1 = new PrivacyProxy();
    const masked = await proxy1.mask('alex@example.com');
    const serialized = proxy1.getMap().serialize();

    const proxy2 = new PrivacyProxy();
    proxy2.loadMap(serialized);
    const restored = proxy2.unmask(masked.text);
    expect(restored).toBe('alex@example.com');
  });

  test('UUID detection and round-trip', async () => {
    const proxy = new PrivacyProxy();
    const original = 'User ID: 550e8400-e29b-41d4-a716-446655440000';

    const masked = await proxy.mask(original);
    expect(masked.text).not.toContain('550e8400-e29b-41d4-a716-446655440000');
    expect(masked.detections[0].type).toBe('uuid');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('credit card detection with Luhn validation', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Card: 4532015112830366';

    const masked = await proxy.mask(original);
    expect(masked.detections.length).toBeGreaterThanOrEqual(1);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('URL with tokens', async () => {
    const proxy = new PrivacyProxy();
    const original = 'Click https://api.example.com/verify?token=abc123secret&user=456';

    const masked = await proxy.mask(original);
    expect(masked.text).not.toContain('abc123secret');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('multiple occurrences of same entity', async () => {
    const proxy = new PrivacyProxy();
    const original = 'From alex@example.com to alex@example.com';

    const masked = await proxy.mask(original);
    const parts = masked.text.split(' to ');
    expect(parts[0].replace('From ', '')).toBe(parts[1]);

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe(original);
  });

  test('custom detector integration', async () => {
    const proxy = new PrivacyProxy({
      detectors: [
        {
          async detect(text) {
            const results: Array<{ type: string; value: string; start: number; end: number }> = [];
            const re = /Patient:\s*([A-Z][a-z]+ [A-Z][a-z]+)/g;
            let m;
            while ((m = re.exec(text)) !== null) {
              results.push({
                type: 'person_name',
                value: m[1],
                start: m.index + m[0].indexOf(m[1]),
                end: m.index + m[0].indexOf(m[1]) + m[1].length,
              });
            }
            return results;
          },
        },
      ],
    });

    const masked = await proxy.mask('Patient: Marcus Weber has an appointment');
    expect(masked.text).not.toContain('Marcus Weber');
    expect(masked.detections[0].type).toBe('person_name');

    const restored = proxy.unmask(masked.text);
    expect(restored).toBe('Patient: Marcus Weber has an appointment');
  });
});
