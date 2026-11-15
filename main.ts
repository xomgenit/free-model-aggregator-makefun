// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone (TypeScript runs directly, and a relative
// import needs its .ts extension). The brief is at the path in $MAKEFUN_PROMPT_FILE. It carries three
// JSON documents: the provider directory, the quality table, and the workload manifest. The env is at
// $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads the brief,
// probes each provider's catalog route, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW A REQUEST IS SERVED
//   The router (lib/router.ts) ranks every free model meeting the request's requirements by the quality
// table's rating, tempered by load and refusals, and this server tries them in order until one answers.
// A provider is always called WITHOUT streaming and its whole answer buffered, then rendered in the
// wire and shape the client asked for (lib/wire.ts): the answer text, usage counts and stop reason pass
// through verbatim; only the envelope changes. A 402, 429 or 5xx is never forwarded: the next candidate
// is tried instead, and the provider that refused is not retried for this request. Exactly one
// successful provider call answers each request.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the providers' answers so far: no wall
// clock, no unseeded randomness. The catalog, the model lists and every rendered answer use fixed ids
// and timestamps. A replay of the trial sees the same provider answers and leaves the same state.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/router.ts      the candidate ranking and the per-provider call and refusal counts.
//   lib/wire.ts        the request translators, the answer parsers and the client renderers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import { Router, type Candidate } from './lib/router.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJsonMode,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, brief.ratings, PROBE_TIMEOUT_MS)));
  for (const probe of probes) {
    if (!probe.ok && probe.error !== null) log(probe.error);
    if (probe.fallback) log(`provider ${probe.provider.id}: catalog from the quality table (live probe failed)`);
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: openAIJsonMode(body) };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface ProviderResult {
  ok: boolean;
  status: number;
  answer: NormalizedAnswer | null;
}

/** One inference call to an OpenAI-wire provider, never streamed. */
async function callOpenAI(provider: ProviderDirectoryEntry, model: string, body: Json, clientApi: 'openai' | 'anthropic'): Promise<ProviderResult> {
  const upstream: Json = clientApi === 'openai' ? { ...body, model, stream: false } : anthropicToOpenAI(body, model);
  delete upstream.stream_options;
  const res = await fetch(`${joinBase(ENV_URL, provider.base)}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) },
    body: JSON.stringify(upstream),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (!res.ok) return { ok: false, status: res.status, answer: null };
  const answer = parseOpenAIAnswer(await res.json().catch(() => null));
  return answer === null ? { ok: false, status: 502, answer: null } : { ok: true, status: 200, answer };
}

/** One inference call to an Anthropic-wire provider, never streamed. */
async function callAnthropic(provider: ProviderDirectoryEntry, model: string, body: Json, clientApi: 'openai' | 'anthropic'): Promise<ProviderResult> {
  const upstream: Json = clientApi === 'anthropic' ? { ...body, model, stream: false } : openAIToAnthropic(body, model);
  const res = await fetch(`${joinBase(ENV_URL, provider.base)}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', ...authHeaders(provider.auth, provider.key) },
    body: JSON.stringify(upstream),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (!res.ok) return { ok: false, status: res.status, answer: null };
  const answer = parseAnthropicAnswer(await res.json().catch(() => null));
  return answer === null ? { ok: false, status: 502, answer: null } : { ok: true, status: 200, answer };
}

/** One inference call to a Gemini-wire provider, never streamed. */
async function callGemini(provider: ProviderDirectoryEntry, model: string, body: Json, clientApi: 'openai' | 'anthropic'): Promise<ProviderResult> {
  const upstream: Json = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const res = await fetch(`${joinBase(ENV_URL, provider.base)}/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) },
    body: JSON.stringify(upstream),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (!res.ok) return { ok: false, status: res.status, answer: null };
  const answer = parseGeminiAnswer(await res.json().catch(() => null));
  return answer === null ? { ok: false, status: 502, answer: null } : { ok: true, status: 200, answer };
}

/** One inference call to a candidate's provider, in that provider's wire. */
async function callCandidate(candidate: Candidate, body: Json, clientApi: 'openai' | 'anthropic'): Promise<ProviderResult> {
  const provider = candidate.probe.provider;
  if (provider.wire === 'openai') return callOpenAI(provider, candidate.model.id, body, clientApi);
  if (provider.wire === 'anthropic') return callAnthropic(provider, candidate.model.id, body, clientApi);
  return callGemini(provider, candidate.model.id, body, clientApi);
}

/**
 * Serve one client request: try the router's candidates in order until one answers. A provider that
 * refuses is recorded and skipped for the rest of this request. Returns the answer and the model that
 * served it, or null when no candidate answered.
 */
async function serve(body: Json, clientApi: 'openai' | 'anthropic'): Promise<{ answer: NormalizedAnswer; model: string } | null> {
  if (router === null) return null;
  const requires = requirementsFor(body);
  // JSON mode is an OpenAI-client property: an Anthropic client cannot ask for it.
  const jsonMode = clientApi === 'openai' && openAIJsonMode(body);
  const tried = new Set<string>();
  for (;;) {
    const candidates = router.candidatesFor(requires, jsonMode, tried);
    if (candidates.length === 0) return null;
    const candidate = candidates[0];
    let result: ProviderResult;
    try {
      result = await callCandidate(candidate, body, clientApi);
    } catch (err) {
      log(`provider ${candidate.probe.provider.id} unreachable: ${(err as Error).message}`);
      result = { ok: false, status: 0, answer: null };
    }
    router.recordCall(candidate.probe.provider.id, !result.ok);
    if (result.ok && result.answer !== null) {
      log(`served by ${candidate.probe.provider.id}/${candidate.model.id} (rating ${candidate.rating})`);
      return { answer: result.answer, model: candidate.model.id };
    }
    log(`provider ${candidate.probe.provider.id}/${candidate.model.id} refused (HTTP ${result.status}); trying the next candidate`);
    tried.add(candidate.probe.provider.id);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered in its wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, 'openai');
  if (served === null) return sendError(res, 'openai', 502, 'upstream_unreachable', 'no free model answered this request');
  if (body.stream === true) {
    const text = renderOpenAIStream(served.answer, served.model);
    res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered in its wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, 'anthropic');
  if (served === null) return sendError(res, 'anthropic', 502, 'api_error', 'no free model answered this request');
  if (body.stream === true) {
    const text = renderAnthropicStream(served.answer, served.model);
    res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
