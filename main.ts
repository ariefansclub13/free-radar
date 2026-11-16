// free-model-aggregator-02: one server, two client wires, one real free model behind every answer.
//
// The trial gives this server a brief (MAKEFUN_PROMPT_FILE): fourteen free OpenRouter models with
// the house's quality ratings and the six prompts' manifest, in the order the env sends them. The
// server listens on 0.0.0.0 at a free port, registers that port and its catalog with the env, and
// answers each prompt by dialling a free model through the grader's gateway on Chat Completions.
// An Anthropic-shaped request is translated into Chat Completions and the answer translated back.
//
// Model choice is a function of the brief, the request and the gateway's answers alone: every prompt
// is served by the best-rated free model meeting its requirements (tools / JSON / vision), with a
// retry on the same model and then failover down the rating order when the gateway refuses a call,
// and a per-model cooldown so one rate-limited model does not take later prompts down with it. No
// wall clock and no unseeded randomness anywhere a request or an answer can see them, so a live pass
// and its replay make the same calls and leave the same state.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Most gateway calls one prompt may spend: 3 candidates, 2 attempts each. */
const MAX_CANDIDATES_PER_PROMPT = 3;
const MAX_ATTEMPTS_PER_CANDIDATE = 2;
/** Most refusal rows the x-makefun-upstream-failures header may carry. */
const MAX_TRAIL_ROWS = 8;
/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's own OpenRouter key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: two JSON documents inside prose, found by their `schema` member.
// ---------------------------------------------------------------------------

interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
const SCHEMA_PREFIX = 'free-model-aggregator-02/';
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!stringOf(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The brief's free models, in the brief's order. */
function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model's capabilities meet one prompt's requirements. Unknown never meets one. */
function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.context_length === null || model.context_length < requires.min_context)) return false;
  return true;
}

/** Free candidates meeting the requirements, best-rated first, brief order breaking ties. */
function rankedCandidates(brief: Brief, requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .map((model, index) => ({ model, index }))
    .filter(({ model }) => meetsRequirements(requires, model))
    .sort((a, b) => b.model.rating_milli - a.model.rating_milli || a.index - b.index)
    .map(({ model }) => model);
}

// ---------------------------------------------------------------------------
// The catalog: the same document --list prints and the register body carries.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogReport {
  providers: Array<{ id: string; base_url: string; key_env: string; wire: 'openai'; models: CatalogModel[] }>;
}

function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [{ id: 'openrouter', base_url: OPENROUTER_PUBLIC_BASE, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models }],
  };
}

// ---------------------------------------------------------------------------
// The client wires: model lists, errors, and Anthropic <-> Chat Completions.
// ---------------------------------------------------------------------------

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** GET /v1/models on an OpenAI-shaped request: OpenAI's list shape. */
function openAIModelList(models: readonly ModelRow[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's list shape. */
function anthropicModelList(models: readonly ModelRow[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A token-count estimate from text length, only when a provider answers with no usage at all. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled. */
function openAIToAnthropic(body: Json, model: string, requestText: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.prompt_tokens);
  const output = numberOrZero(usage.completion_tokens);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: input > 0 ? input : estimateTokens(requestText),
      output_tokens: output > 0 ? output : estimateTokens(text),
    },
  };
}

/** A Chat Completions answer with `model` set to the id dialled and usage always whole. */
function openAIJsonWithModel(body: Json, model: string, requestText: string): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(out.choices) ? out.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  if (typeof choice.finish_reason !== 'string') {
    out.choices = [{ ...choice, finish_reason: 'stop' }, ...choices.slice(1)];
  }
  const usage = isRecord(out.usage) ? out.usage : {};
  const input = numberOrZero(usage.prompt_tokens);
  const output = numberOrZero(usage.completion_tokens);
  const text = typeof choice.message === 'object' && choice.message !== null ? textOf((choice.message as Json).content) : '';
  out.usage = {
    prompt_tokens: input > 0 ? input : estimateTokens(requestText),
    completion_tokens: output > 0 ? output : estimateTokens(text),
    total_tokens: (input > 0 ? input : estimateTokens(requestText)) + (output > 0 ? output : estimateTokens(text)),
  };
  return out;
}

// ---------------------------------------------------------------------------
// SSE: one event, and a buffered upstream Chat Completions stream, validated.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split SSE text into frames (each frame's raw text, without the blank line that ends it). */
function sseFrames(text: string): string[] {
  const frames: string[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  let rest = normalized;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    frames.push(rest.slice(0, at));
    rest = rest.slice(at + 2);
    at = rest.indexOf('\n\n');
  }
  if (rest.trim() !== '') frames.push(rest);
  return frames;
}

/** One frame's data payloads, each parsed when it is JSON. */
function framePayloads(frame: string): Array<{ raw: string; value: unknown }> {
  const out: Array<{ raw: string; value: unknown }> = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const raw = trimmed.slice(5).trim();
    if (raw === '') continue;
    let value: unknown = raw;
    try {
      value = JSON.parse(raw);
    } catch {
      // Not JSON: keep the raw payload.
    }
    out.push({ raw, value });
  }
  return out;
}

interface CollectedStream {
  /** The upstream frames, each data payload's `model` set to the id dialled. */
  frames: string[];
  /** The answer's text, in order. */
  text: string;
  /** The stop reason, or null when no chunk carried one. */
  finishReason: string | null;
  /** The usage the stream carried, or null. */
  usage: { input: number; output: number } | null;
  /** Tool calls, in order, with their arguments accumulated. */
  toolCalls: Array<{ id: string; name: string; arguments: string }>;
  /** Whether the stream ended with [DONE]. */
  sawDone: boolean;
}

/**
 * Collect an upstream Chat Completions stream whole, replacing every chunk's `model` with the id
 * dialled. Collecting (rather than relaying) is what lets the Anthropic wire be built with its true
 * usage in message_start, and lets a stream that never finishes be treated as a failed call instead
 * of a broken answer. Nothing here depends on timing, only on the frames.
 */
async function collectStream(upstream: ReadableStream<Uint8Array>, model: string): Promise<CollectedStream> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    }
    buffer += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  const collected: CollectedStream = { frames: [], text: '', finishReason: null, usage: null, toolCalls: [], sawDone: false };
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const frame of sseFrames(buffer)) {
    const rewritten: string[] = [];
    let isDoneFrame = false;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) {
        rewritten.push(line);
        continue;
      }
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        isDoneFrame = true;
        rewritten.push(line);
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        rewritten.push(line);
        continue;
      }
      if (!isRecord(chunk)) {
        rewritten.push(line);
        continue;
      }
      if (isRecord(chunk.usage)) {
        collected.usage = {
          input: numberOrZero(chunk.usage.prompt_tokens),
          output: numberOrZero(chunk.usage.completion_tokens),
        };
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null) {
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') collected.text += delta.content;
        const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call)) continue;
          const index = typeof call.index === 'number' ? call.index : byIndex.size;
          let entry = byIndex.get(index);
          if (entry === undefined) {
            entry = { id: '', name: '', arguments: '' };
            byIndex.set(index, entry);
          }
          const fn = isRecord(call.function) ? call.function : {};
          if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
          if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
          if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
        }
        if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
          collected.finishReason = choice.finish_reason;
        }
      }
      rewritten.push(`data: ${JSON.stringify({ ...chunk, model })}`);
    }
    collected.frames.push(rewritten.join('\n'));
    if (isDoneFrame) collected.sawDone = true;
  }
  collected.toolCalls = [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, entry]) => entry);
  return collected;
}

/**
 * A collected Chat Completions stream as the Anthropic event sequence: message_start (with the true
 * input count, known because the stream was collected whole), one text block for the text, one tool_use
 * block per tool call, then message_delta with the stop reason and the output count, then message_stop.
 */
function anthropicEvents(collected: CollectedStream, model: string, requestText: string): string[] {
  const out: string[] = [];
  const input = collected.usage !== null && collected.usage.input > 0 ? collected.usage.input : estimateTokens(requestText);
  const output = collected.usage !== null && collected.usage.output > 0 ? collected.usage.output : estimateTokens(collected.text);
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (collected.text !== '') {
    out.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: collected.text } }));
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of collected.toolCalls) {
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(collected.finishReason), stop_sequence: null },
      usage: { output_tokens: output },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out;
}

// ---------------------------------------------------------------------------
// The gateway: one dial per model, with retry, failover and the refusal trail.
// ---------------------------------------------------------------------------

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/** Models that refused a call this trial and are cooling down for later prompts. */
const cooledDown = new Set<string>();

/** The gateway's Chat Completions route. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The error code of a refused gateway answer: its error.code, else its error.type, else ''. */
async function failureCode(res: Response): Promise<string> {
  let text = '';
  try {
    text = await res.text();
  } catch {
    return '';
  }
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code.
  }
  return '';
}

/** One POST to the gateway's Chat Completions route, naming the model dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

type DialOutcome =
  | { ok: true; model: string; response: Response; failures: FailureRow[] }
  | { ok: false; status: number; code: string; failures: FailureRow[] };

/**
 * Serve one Chat Completions body through the gateway. Candidates are the brief's free models
 * meeting the requirements, best-rated first; models that refused earlier in the trial are skipped
 * while a fresh candidate remains. Each candidate is tried twice on a 429 or 5xx, then the next
 * candidate is tried, at most three candidates. A 401 (no key pushed) or any other 4xx is a
 * request-side refusal: it is not retried, and no other model can serve it either. Every refusal is
 * kept for the x-makefun-upstream-failures trail, in the order the calls refused.
 */
async function dialWithFailover(openaiBody: Json, requires: Requirement): Promise<DialOutcome> {
  const ranked = rankedCandidates(brief, requires);
  if (ranked.length === 0) return { ok: false, status: 503, code: 'no_free_model', failures: [] };
  const fresh = ranked.filter((m) => !cooledDown.has(m.id));
  const order = (fresh.length > 0 ? fresh : ranked).slice(0, MAX_CANDIDATES_PER_PROMPT);
  const failures: FailureRow[] = [];
  for (const candidate of order) {
    for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_CANDIDATE; attempt++) {
      let upstream: Response;
      try {
        upstream = await dial(candidate.id, openaiBody);
      } catch (err) {
        failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
        cooledDown.add(candidate.id);
        break;
      }
      if (upstream.ok && upstream.body !== null) {
        return { ok: true, model: candidate.id, response: upstream, failures };
      }
      const code = await failureCode(upstream);
      failures.push({ status: upstream.status, code, model: candidate.id });
      cooledDown.add(candidate.id);
      if (upstream.status === 401) return { ok: false, status: 401, code, failures };
      // A 403/404 is final for this model (key fine, model refuses us: e.g. a
      // free model gated to agentic harnesses) — do not retry it, try the next
      // candidate instead of losing the prompt.
      if (upstream.status === 403 || upstream.status === 404) break;
      if (upstream.status !== 429 && upstream.status < 500) return { ok: false, status: upstream.status, code, failures };
      // A 429 or 5xx: one retry on this same model, then the next candidate.
    }
  }
  const last = failures[failures.length - 1];
  return { ok: false, status: 502, code: last !== undefined && last.code !== '' ? last.code : 'upstream_error', failures };
}

/** The failures trail as one header value, or null when nothing refused. */
function trailHeader(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS));
}

// ---------------------------------------------------------------------------
// HTTP: one client request body, the routes, and the answers.
// ---------------------------------------------------------------------------

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

function sendJson(res: ServerResponse, status: number, body: unknown, trail: string | null = null): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, code: string | null = null): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, code));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = freeModels(brief);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', outcome: Extract<DialOutcome, { ok: false }>): void {
  const message = `the gateway answered HTTP ${outcome.status}${outcome.code === '' ? '' : `: ${outcome.code}`}`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', message);
  else sendError(res, wire, 502, 'upstream_error', message, outcome.code === '' ? null : outcome.code);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a real free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const requestText = JSON.stringify(body);
  const outcome = await dialWithFailover({ ...body, model: 'auto' }, requires);
  if (!outcome.ok) return refused(res, 'openai', outcome);
  const trail = trailHeader(outcome.failures);
  if (body.stream === true) {
    const collected = await collectStream(outcome.response.body as ReadableStream<Uint8Array>, outcome.model);
    if (collected.finishReason === null) {
      // The upstream stream never finished: a broken answer is not an answer. Fail the prompt
      // rather than serve a stream the env's order check would refuse.
      return refused(res, 'openai', { ok: false, status: 502, code: 'upstream_incomplete_stream', failures: outcome.failures });
    }
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
    res.writeHead(200, headers);
    for (const frame of collected.frames) res.write(`${frame}\n\n`);
    if (!collected.sawDone) res.write('data: [DONE]\n\n');
    res.end();
    return;
  }
  let answer: unknown;
  try {
    answer = await outcome.response.json();
  } catch {
    return refused(res, 'openai', { ok: false, status: 502, code: 'upstream_bad_json', failures: outcome.failures });
  }
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, outcome.model, requestText), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const requestText = JSON.stringify(body);
  const outcome = await dialWithFailover(anthropicToOpenAI(body, 'auto'), requires);
  if (!outcome.ok) return refused(res, 'anthropic', outcome);
  const trail = trailHeader(outcome.failures);
  if (body.stream === true) {
    const collected = await collectStream(outcome.response.body as ReadableStream<Uint8Array>, outcome.model);
    if (collected.finishReason === null) {
      return refused(res, 'anthropic', { ok: false, status: 502, code: 'upstream_incomplete_stream', failures: outcome.failures });
    }
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
    res.writeHead(200, headers);
    for (const event of anthropicEvents(collected, outcome.model, requestText)) res.write(event);
    res.end();
    return;
  }
  let answer: unknown;
  try {
    answer = await outcome.response.json();
  } catch {
    return refused(res, 'anthropic', { ok: false, status: 502, code: 'upstream_bad_json', failures: outcome.failures });
  }
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, outcome.model, requestText), trail);
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

// ---------------------------------------------------------------------------
// Start: listen, register with the env, poll /work, exit 0 when it says done.
// ---------------------------------------------------------------------------

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog, retrying while the env is not yet up. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 30; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
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
  log(`listening on 0.0.0.0:${port} with ${freeModels(brief).length} free models`);
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
