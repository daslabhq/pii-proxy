/**
 * Bijective map — deterministic, reversible mapping between real and fake values.
 *
 * Same real value always maps to the same fake value within a session.
 * The reverse is also true — fake → real is unambiguous.
 */
export class BijectiveMap {
  private realToFake = new Map<string, string>();
  private fakeToReal = new Map<string, string>();
  private types = new Map<string, string>();
  private counts = new Map<string, number>();

  /** Map a real value to its fake replacement, optionally recording what kind of value it is. */
  set(real: string, fake: string, type?: string): void {
    this.realToFake.set(real, fake);
    this.fakeToReal.set(fake, real);
    if (!type) return;
    const before = this.types.get(real);
    if (before === type) return;
    if (before) this.counts.set(before, (this.counts.get(before) ?? 1) - 1);
    this.counts.set(type, (this.counts.get(type) ?? 0) + 1);
    this.types.set(real, type);
  }

  /** The type a real value was masked as, when it was recorded. */
  typeOf(real: string): string | undefined {
    return this.types.get(real);
  }

  /** How many mapped values have this type. */
  countOf(type: string): number {
    return this.counts.get(type) ?? 0;
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

  /** Serialize the map for persistence (e.g., Redis, SQLite). Typed entries carry their type third. */
  serialize(): string {
    const entries = Array.from(this.realToFake.entries(), ([real, fake]) => {
      const type = this.types.get(real);
      return type ? [real, fake, type] : [real, fake];
    });
    return JSON.stringify(entries);
  }

  /** Restore from serialized data. */
  static deserialize(data: string): BijectiveMap {
    const map = new BijectiveMap();
    const entries: Array<[string, string, string?]> = JSON.parse(data);
    for (const [real, fake, type] of entries) {
      map.set(real, fake, type);
    }
    return map;
  }

  /** Iterate all mappings (for debugging / export). */
  entries(): IterableIterator<[string, string]> {
    return this.realToFake.entries();
  }
}
