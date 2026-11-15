// Routing: which free model serves one request, and what to try when a provider refuses.
//
// Every request is served by the best free model that meets its requirements, where "best" is the
// quality table's rating_milli, tempered by what the trial has taught so far:
//
//   score = rating_milli - 2 * callsMade(provider) - 3 * refusals(provider)
//
// The penalties stay below the rating gaps between the top providers (5 points): one refusal or a
// few calls never demote a 1475 model below a fresh 1470 one. The per-call penalty spreads load
// across providers at near-equal ratings (a fresh 1470 beats a three-times-used 1475), which matters
// because a provider's first call always succeeds — its failure schedule and its outage both start
// at call 2 — while later calls may refuse. The refusal penalty routes around a provider that just
// refused, without blacklisting it for one transient failure-schedule hit. A provider with three
// refusals is dead: at most two of them can be its failure schedule, so the third means its budget
// is spent or its outage began, and both refuse forever.
//
// Within one request, a provider that refused is not retried: the next candidate is tried instead.
// Retrying costs an extra upstream call (the tie-break counts them) for a few rating points, and the
// next-best provider is only a few points lower. A request stops at its first success: exactly one
// successful provider call may answer it.
//
// A JSON-mode request from an OpenAI client is never routed to an Anthropic-wire provider: that wire
// cannot carry response_format, so the provider would echo json_mode false and the request would fail
// its echo check. The Gemini wire carries it as responseMimeType.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingMilliOf,
  ratingOfModel,
  type Capabilities,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type Wire,
} from './providers.ts';

/** One (provider, model) pair that could serve a request, with its effective score. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
}

/** How much one call already made to a provider costs its models' scores. */
const CALL_PENALTY = 2;
/** How much one refusal from a provider costs its models' scores. */
const REFUSAL_PENALTY = 3;
/** Refusals after which a provider is treated as exhausted or out, for the rest of the trial. */
const DEAD_AFTER_REFUSALS = 3;
/** The score penalty of a dead provider: below every live candidate. */
const DEAD_PENALTY = 100000;

export class Router {
  private readonly probes: Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly calls = new Map<string, number>();
  private readonly refusals = new Map<string, number>();

  constructor(probes: Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
    for (const probe of probes) {
      this.calls.set(probe.provider.id, 0);
      this.refusals.set(probe.provider.id, 0);
    }
  }

  /** How many calls this trial has made to a provider, successes and refusals alike. */
  callsMade(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  /** How many times a provider has refused a call this trial. */
  refusalsOf(providerId: string): number {
    return this.refusals.get(providerId) ?? 0;
  }

  /** Whether a provider is treated as dead: three refusals prove its budget is spent or it is out. */
  isDead(providerId: string): boolean {
    return (this.refusals.get(providerId) ?? 0) >= DEAD_AFTER_REFUSALS;
  }

  /** Record one provider call's outcome. A refusal is a 402, a 429 or a 5xx (or a transport failure). */
  recordCall(providerId: string, refused: boolean): void {
    this.calls.set(providerId, (this.calls.get(providerId) ?? 0) + 1);
    if (refused) this.refusals.set(providerId, (this.refusals.get(providerId) ?? 0) + 1);
  }

  /**
   * The candidates for one request, best first: free models meeting the requirements, scored by
   * rating less the load and refusal penalties. `jsonMode` is true when an OpenAI client asked for
   * JSON mode, which excludes Anthropic-wire providers. `exclude` holds the providers that already
   * refused this request. The order is total and deterministic: score, then rating, then provider id,
   * then model id.
   *
   * The penalties stay smaller than the rating gaps between the top providers (5 points): one
   * refusal (-3) or a couple of calls (-2 each) never demotes a 1475 model below a fresh 1470 one. A
   * single refusal is usually a transient failure-schedule hit, not exhaustion, so it must not
   * blacklist a provider for the rest of the trial; the `exclude` set already routes around a
   * refusal within one request, and the dead rule (three refusals) catches a provider whose budget
   * is spent or whose outage began.
   */
  candidatesFor(requires: Requirement, jsonMode: boolean, exclude: ReadonlySet<string>): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      if (exclude.has(probe.provider.id)) continue;
      if (jsonMode && probe.provider.wire === 'anthropic') continue;
      const wire: Wire = probe.provider.wire;
      void wire;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        const caps: Capabilities = capabilitiesOf(model, row);
        if (!meetsRequirements(requires, caps)) continue;
        const rating = ratingMilliOf(this.ratings, probe.provider.id, model.id);
        const calls = this.callsMade(probe.provider.id);
        const refusals = this.refusalsOf(probe.provider.id);
        const score = rating - CALL_PENALTY * calls - REFUSAL_PENALTY * refusals - (this.isDead(probe.provider.id) ? DEAD_PENALTY : 0);
        out.push({ probe, model, rating, score });
      }
    }
    out.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    return out;
  }
}
