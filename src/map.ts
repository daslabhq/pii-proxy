/**
 * Bijective map — deterministic, reversible mapping between real and fake values.
 *
 * Same real value always maps to the same fake value within a session.
 * The reverse is also true — fake → real is unambiguous.
 */
export class BijectiveMap {
  private realToFake = new Map<string, string>();
  private fakeToReal = new Map<string, string>();

  /** Map a real value to its fake replacement, or return existing mapping. */
  set(real: string, fake: string): void {
    this.realToFake.set(real, fake);
    this.fakeToReal.set(fake, real);
  }

  /** Get the fake value for a real value, or undefined. */
  getFake(real: string): string | undefined {
    return this.realToFake.get(real);
  }

  /** Get the real value for a fake value, or undefined. */
  getReal(fake: string): string | undefined {
    return this.fakeToReal.get(fake);
  }

  /** Check if a real value has been mapped. */
  hasFake(real: string): boolean {
    return this.realToFake.has(real);
  }

  get size(): number {
    return this.realToFake.size;
  }

  /** Serialize the map for persistence (e.g., Redis, SQLite). */
  serialize(): string {
    const entries = Array.from(this.realToFake.entries());
    return JSON.stringify(entries);
  }

  /** Restore from serialized data. */
  static deserialize(data: string): BijectiveMap {
    const map = new BijectiveMap();
    const entries: [string, string][] = JSON.parse(data);
    for (const [real, fake] of entries) {
      map.set(real, fake);
    }
    return map;
  }

  /** Iterate all mappings (for debugging / export). */
  entries(): IterableIterator<[string, string]> {
    return this.realToFake.entries();
  }
}
