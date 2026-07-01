# Mixture of Agents (MoA) Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a virtual `moa` model provider to cz-cli where a selected preset runs reference models in parallel for advisory context, then hands off to an aggregator that acts as the normal tool-calling model.

**Architecture:** Approach A2 — selection flows through the existing model system (a synthesized `moa` provider whose models are presets), while the reference fan-out + tool-aware context injection happen in `session/llm.ts`'s `run()` right before `streamText`. Pure preset/message logic lives in a dedicated, unit-testable module; the runtime wiring lives in `llm.ts`. The aggregator's language model is what actually streams, so tool schema, tool execution, iteration, and prompt caching are all unchanged.

**Tech Stack:** TypeScript, Effect, Vercel AI SDK (`ai@6.0.158` — `generateText`/`streamText`), zod (config), `bun test` (`bun:test`).

## Global Constraints

- Repo: `/Users/guanyangw/cz-cli`, branch `moa-port`. All work stays on this branch.
- Package: everything is under `packages/opencode/`. Test runner: `bun test --timeout 30000` (run from `packages/opencode/`).
- Model strings are `provider/model`, split on the FIRST `/` via `parseModel` (`packages/opencode/src/provider/provider.ts:1878`): `"openrouter/deepseek/deepseek-v4-pro"` → provider `openrouter`, model `deepseek/deepseek-v4-pro`.
- Tests use `import { test, expect } from "bun:test"` and Effect runtime helpers already established in `packages/opencode/test/`.
- No hardcoded `max_tokens` default (unset ⇒ model's own maximum). No per-preset temperature fields.
- Reference concurrency default is `8`, overridable via `moa.reference_concurrency`.
- Recursion: a preset's `aggregator` or any `reference_models` entry must NEVER be a `moa/*` model — reject at config layer, re-assert at runtime.
- Reference display name prefix: `MoA: <preset>`.
- Spec: `docs/superpowers/specs/2026-07-01-moa-mixture-of-agents-port-design.md`.

## File Structure

- **Create** `packages/opencode/src/session/moa.ts` — pure MoA logic module (no I/O, no Effect): preset normalization/resolution, recursion guards, reference-message filtering, tool-list text, context synthesis + injection. This is the heavily unit-tested core (mirrors hermes `moa_config.py` + the pure parts of `moa_loop.py`).
- **Modify** `packages/opencode/src/config/config.ts` — add the `moa` block to the `Info` zod schema (~line 93).
- **Modify** `packages/opencode/src/provider/provider.ts` — synthesize the `moa` virtual provider during provider assembly; guard `getLanguage` against `moa/*`.
- **Modify** `packages/opencode/src/session/llm.ts` — in `run()`, detect `providerID === "moa"`, run the reference fan-out (via `moa.ts`), inject context, and swap in the aggregator's language model for `streamText`.
- **Create** `packages/opencode/test/session/moa.test.ts` — unit tests for `moa.ts`.
- **Create** `packages/opencode/test/provider/moa-provider.test.ts` — tests for virtual provider registration + capability inheritance.
- **Create** `packages/opencode/test/session/moa-integration.test.ts` — real multi-model E2E, gated behind an env var (skipped by default).

---

### Task 1: MoA config types + preset normalization (pure)

**Files:**
- Create: `packages/opencode/src/session/moa.ts`
- Test: `packages/opencode/test/session/moa.test.ts`

**Interfaces:**
- Consumes: nothing (pure module, standalone).
- Produces:
  - `type MoASlot = { providerID: string; modelID: string }`
  - `type MoAPreset = { enabled: boolean; reference_models: MoASlot[]; aggregator: MoASlot; max_tokens?: number }`
  - `type MoAConfig = { default_preset: string; reference_concurrency: number; presets: Record<string, MoAPreset> }`
  - `function parseSlot(model: string): MoASlot` — splits on first `/` (same rule as `parseModel`).
  - `function normalizeMoAConfig(raw: unknown): MoAConfig` — validates/normalizes raw config into `MoAConfig`; drops `moa/*` slots (recursion guard); a preset whose `aggregator` resolves to `moa/*` or is missing is dropped; `reference_concurrency` defaults to `8`; `default_preset` falls back to the first preset name.
  - `function resolveMoAPreset(cfg: MoAConfig, name?: string): MoAPreset` — returns the named preset (or default); throws `Error` naming available presets if not found.
  - `const DEFAULT_REFERENCE_CONCURRENCY = 8`

- [ ] **Step 1: Write the failing test for `parseSlot`**

Create `packages/opencode/test/session/moa.test.ts`:

```ts
import { test, expect, describe } from "bun:test"
import { parseSlot, normalizeMoAConfig, resolveMoAPreset, DEFAULT_REFERENCE_CONCURRENCY } from "../../src/session/moa"

describe("parseSlot", () => {
  test("splits provider from model on first slash", () => {
    expect(parseSlot("anthropic/claude-opus-4.8")).toEqual({ providerID: "anthropic", modelID: "claude-opus-4.8" })
  })
  test("keeps later slashes in the model id", () => {
    expect(parseSlot("openrouter/deepseek/deepseek-v4-pro")).toEqual({
      providerID: "openrouter",
      modelID: "deepseek/deepseek-v4-pro",
    })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `packages/opencode/`): `bun test test/session/moa.test.ts`
Expected: FAIL — cannot resolve `../../src/session/moa`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/opencode/src/session/moa.ts`:

```ts
export const DEFAULT_REFERENCE_CONCURRENCY = 8

export type MoASlot = { providerID: string; modelID: string }
export type MoAPreset = {
  enabled: boolean
  reference_models: MoASlot[]
  aggregator: MoASlot
  max_tokens?: number
}
export type MoAConfig = {
  default_preset: string
  reference_concurrency: number
  presets: Record<string, MoAPreset>
}

export function parseSlot(model: string): MoASlot {
  const [providerID, ...rest] = model.split("/")
  return { providerID, modelID: rest.join("/") }
}

function cleanSlot(model: unknown): MoASlot | undefined {
  if (typeof model !== "string" || !model.includes("/")) return undefined
  const slot = parseSlot(model)
  if (!slot.providerID || !slot.modelID) return undefined
  if (slot.providerID.toLowerCase() === "moa") return undefined // recursion guard
  return slot
}

export function normalizeMoAConfig(raw: unknown): MoAConfig {
  const obj = (raw && typeof raw === "object" ? raw : {}) as Record<string, any>
  const presets: Record<string, MoAPreset> = {}
  const rawPresets = obj.presets && typeof obj.presets === "object" ? obj.presets : {}
  for (const [name, p] of Object.entries<any>(rawPresets)) {
    const cleanName = String(name || "").trim()
    if (!cleanName) continue
    const aggregator = cleanSlot(p?.aggregator)
    if (!aggregator) continue // no acting model ⇒ drop preset
    const refs = (Array.isArray(p?.reference_models) ? p.reference_models : [])
      .map(cleanSlot)
      .filter((s: MoASlot | undefined): s is MoASlot => s !== undefined)
    if (refs.length === 0) continue // classic MoA needs ≥1 reference
    const maxTokens =
      typeof p?.max_tokens === "number" && p.max_tokens > 0 ? Math.floor(p.max_tokens) : undefined
    presets[cleanName] = {
      enabled: p?.enabled !== false,
      reference_models: refs,
      aggregator,
      max_tokens: maxTokens,
    }
  }
  const concurrency =
    typeof obj.reference_concurrency === "number" && obj.reference_concurrency > 0
      ? Math.floor(obj.reference_concurrency)
      : DEFAULT_REFERENCE_CONCURRENCY
  let defaultName = String(obj.default_preset || "").trim()
  if (!defaultName || !presets[defaultName]) defaultName = Object.keys(presets)[0] ?? ""
  return { default_preset: defaultName, reference_concurrency: concurrency, presets }
}

export function resolveMoAPreset(cfg: MoAConfig, name?: string): MoAPreset {
  const presetName = String(name || cfg.default_preset || "").trim()
  const preset = cfg.presets[presetName]
  if (!preset) {
    const available = Object.keys(cfg.presets).join(", ") || "(none)"
    throw new Error(`MoA preset "${presetName}" not found. Available presets: ${available}`)
  }
  return preset
}
```

- [ ] **Step 4: Run the `parseSlot` test to verify it passes**

Run: `bun test test/session/moa.test.ts`
Expected: PASS.

- [ ] **Step 5: Add normalization + resolution tests**

Append to `packages/opencode/test/session/moa.test.ts`:

```ts
describe("normalizeMoAConfig", () => {
  const good = {
    presets: {
      default: {
        reference_models: ["openai/gpt-5.5", "openrouter/deepseek/deepseek-v4-pro"],
        aggregator: "anthropic/claude-opus-4.8",
      },
    },
  }

  test("normalizes a valid preset", () => {
    const cfg = normalizeMoAConfig(good)
    expect(cfg.default_preset).toBe("default")
    expect(cfg.reference_concurrency).toBe(DEFAULT_REFERENCE_CONCURRENCY)
    expect(cfg.presets.default.aggregator).toEqual({ providerID: "anthropic", modelID: "claude-opus-4.8" })
    expect(cfg.presets.default.reference_models).toHaveLength(2)
    expect(cfg.presets.default.enabled).toBe(true)
  })

  test("drops moa/* recursion slots (aggregator)", () => {
    const cfg = normalizeMoAConfig({
      presets: { p: { reference_models: ["openai/gpt-5.5"], aggregator: "moa/other" } },
    })
    expect(cfg.presets.p).toBeUndefined()
  })

  test("drops moa/* reference slots but keeps valid ones", () => {
    const cfg = normalizeMoAConfig({
      presets: { p: { reference_models: ["moa/x", "openai/gpt-5.5"], aggregator: "anthropic/claude-opus-4.8" } },
    })
    expect(cfg.presets.p.reference_models).toEqual([{ providerID: "openai", modelID: "gpt-5.5" }])
  })

  test("drops a preset with no valid references", () => {
    const cfg = normalizeMoAConfig({
      presets: { p: { reference_models: ["moa/x"], aggregator: "anthropic/claude-opus-4.8" } },
    })
    expect(cfg.presets.p).toBeUndefined()
  })

  test("respects enabled:false and custom concurrency + max_tokens", () => {
    const cfg = normalizeMoAConfig({
      reference_concurrency: 3,
      presets: {
        p: {
          enabled: false,
          reference_models: ["openai/gpt-5.5"],
          aggregator: "anthropic/claude-opus-4.8",
          max_tokens: 2048,
        },
      },
    })
    expect(cfg.reference_concurrency).toBe(3)
    expect(cfg.presets.p.enabled).toBe(false)
    expect(cfg.presets.p.max_tokens).toBe(2048)
  })

  test("default_preset falls back to first preset when unset/invalid", () => {
    const cfg = normalizeMoAConfig({ default_preset: "nope", presets: (good as any).presets })
    expect(cfg.default_preset).toBe("default")
  })

  test("bad input degrades to empty presets", () => {
    expect(normalizeMoAConfig(null).presets).toEqual({})
    expect(normalizeMoAConfig(42 as any).presets).toEqual({})
  })
})

describe("resolveMoAPreset", () => {
  test("resolves default when name omitted", () => {
    const cfg = normalizeMoAConfig({
      presets: { default: { reference_models: ["openai/gpt-5.5"], aggregator: "anthropic/claude-opus-4.8" } },
    })
    expect(resolveMoAPreset(cfg).aggregator.modelID).toBe("claude-opus-4.8")
  })
  test("throws with available names when preset missing", () => {
    const cfg = normalizeMoAConfig({
      presets: { default: { reference_models: ["openai/gpt-5.5"], aggregator: "anthropic/claude-opus-4.8" } },
    })
    expect(() => resolveMoAPreset(cfg, "ghost")).toThrow(/not found.*default/)
  })
})
```

- [ ] **Step 6: Run all `moa.test.ts` tests**

Run: `bun test test/session/moa.test.ts`
Expected: PASS (all describes).

- [ ] **Step 7: Commit**

```bash
git add packages/opencode/src/session/moa.ts packages/opencode/test/session/moa.test.ts
git commit -m "feat(moa): add pure preset config normalization"
```

---

### Task 2: Reference message view, tool-list text, context injection (pure)

**Files:**
- Modify: `packages/opencode/src/session/moa.ts`
- Test: `packages/opencode/test/session/moa.test.ts`

**Interfaces:**
- Consumes: `MoASlot` from Task 1; `ModelMessage` from `ai`.
- Produces:
  - `function referenceMessages(messages: ModelMessage[]): ModelMessage[]` — keeps only user/assistant TEXT turns; drops system, `tool`-role, and assistant turns that carry no text (pure tool calls).
  - `function toolListText(tools: { name: string; description?: string }[]): string` — renders a short bullet list of available tools for the reference system prompt (empty string if no tools).
  - `function referenceSystemPrompt(toolList: string): string` — advisory framing + tool list, used as the reference call's `system`.
  - `function synthesizeContext(input: { preset: string; aggregatorLabel: string; referenceLabels: string[]; outputs: { label: string; text: string }[] }): string` — builds the `[Mixture of Agents reference context]` block.
  - `function injectContext(messages: ModelMessage[], context: string): ModelMessage[]` — appends `context` to the tail of the LAST user message's text; if no user message exists, appends a new user message. Returns a new array (does not mutate input).

- [ ] **Step 1: Write failing tests**

Append to `packages/opencode/test/session/moa.test.ts`:

```ts
import type { ModelMessage } from "ai"
import { referenceMessages, toolListText, referenceSystemPrompt, synthesizeContext, injectContext } from "../../src/session/moa"

describe("referenceMessages", () => {
  test("keeps user/assistant text, drops system and tool roles", () => {
    const msgs: ModelMessage[] = [
      { role: "system", content: "SYS" },
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "1", toolName: "x", output: { type: "text", value: "r" } }] } as any,
    ]
    const out = referenceMessages(msgs)
    expect(out).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
    ])
  })

  test("drops assistant turns that are pure tool calls (no text)", () => {
    const msgs: ModelMessage[] = [
      { role: "user", content: "do it" },
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "1", toolName: "bash", input: {} }] } as any,
    ]
    expect(referenceMessages(msgs)).toEqual([{ role: "user", content: "do it" }])
  })

  test("extracts text parts from array assistant content", () => {
    const msgs: ModelMessage[] = [
      { role: "assistant", content: [{ type: "text", text: "thinking" }, { type: "tool-call", toolCallId: "1", toolName: "b", input: {} }] } as any,
    ]
    expect(referenceMessages(msgs)).toEqual([{ role: "assistant", content: "thinking" }])
  })
})

describe("toolListText / referenceSystemPrompt", () => {
  test("renders bullet list of tools", () => {
    const t = toolListText([{ name: "bash", description: "Run a shell command" }, { name: "read" }])
    expect(t).toContain("bash")
    expect(t).toContain("Run a shell command")
    expect(t).toContain("read")
  })
  test("empty tools ⇒ empty string", () => {
    expect(toolListText([])).toBe("")
  })
  test("system prompt embeds the tool list", () => {
    const sys = referenceSystemPrompt(toolListText([{ name: "bash", description: "shell" }]))
    expect(sys).toContain("bash")
    expect(sys.toLowerCase()).toContain("tool")
  })
})

describe("synthesizeContext / injectContext", () => {
  const ctx = synthesizeContext({
    preset: "default",
    aggregatorLabel: "anthropic/claude-opus-4.8",
    referenceLabels: ["openai/gpt-5.5"],
    outputs: [{ label: "openai/gpt-5.5", text: "use grep first" }],
  })

  test("context block contains preset, aggregator, and reference output", () => {
    expect(ctx).toContain("Mixture of Agents reference context")
    expect(ctx).toContain("default")
    expect(ctx).toContain("anthropic/claude-opus-4.8")
    expect(ctx).toContain("Reference 1")
    expect(ctx).toContain("use grep first")
  })

  test("injects at tail of last user message", () => {
    const msgs: ModelMessage[] = [
      { role: "user", content: "first" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "second" },
    ]
    const out = injectContext(msgs, "CTX")
    expect(out[2].content).toBe("second\n\nCTX")
    expect(msgs[2].content).toBe("second") // original not mutated
  })

  test("appends a user message when none present", () => {
    const out = injectContext([{ role: "assistant", content: "x" }], "CTX")
    expect(out[out.length - 1]).toEqual({ role: "user", content: "CTX" })
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `bun test test/session/moa.test.ts`
Expected: FAIL — new functions not exported.

- [ ] **Step 3: Implement in `moa.ts`**

Append to `packages/opencode/src/session/moa.ts`:

```ts
import type { ModelMessage } from "ai"

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return ""
  return content
    .filter((p: any) => p && p.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text)
    .join("")
}

export function referenceMessages(messages: ModelMessage[]): ModelMessage[] {
  const out: ModelMessage[] = []
  for (const msg of messages) {
    if (msg.role !== "user" && msg.role !== "assistant") continue
    const text = textFromContent(msg.content)
    if (!text.trim()) continue
    out.push({ role: msg.role, content: text })
  }
  return out
}

export function toolListText(tools: { name: string; description?: string }[]): string {
  if (tools.length === 0) return ""
  return tools
    .map((t) => (t.description ? `- ${t.name}: ${t.description.split("\n")[0]}` : `- ${t.name}`))
    .join("\n")
}

export function referenceSystemPrompt(toolList: string): string {
  const base =
    "You are a reference model in a Mixture of Agents process. Give concise, actionable advice " +
    "for the acting agent: next steps, tool-use strategy, risks. You cannot call tools yourself; " +
    "describe the strategy in text."
  if (!toolList) return base
  return `${base}\n\nThe acting agent has these tools available:\n${toolList}`
}

export function synthesizeContext(input: {
  preset: string
  aggregatorLabel: string
  referenceLabels: string[]
  outputs: { label: string; text: string }[]
}): string {
  const joined = input.outputs
    .map((o, i) => `Reference ${i + 1} — ${o.label}:\n${o.text}`)
    .join("\n\n")
  return (
    "[Mixture of Agents reference context]\n" +
    `Preset: ${input.preset}\n` +
    `Aggregator/acting model: ${input.aggregatorLabel}\n` +
    `References: ${input.referenceLabels.join(", ")}\n\n` +
    "Use the reference responses below as private context. You are the aggregator and acting model: " +
    "answer the user directly or call tools as needed.\n\n" +
    joined
  )
}

export function injectContext(messages: ModelMessage[], context: string): ModelMessage[] {
  const out = messages.map((m) => ({ ...m }))
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i].role === "user" && typeof out[i].content === "string") {
      out[i] = { ...out[i], content: `${out[i].content}\n\n${context}` }
      return out
    }
  }
  out.push({ role: "user", content: context })
  return out
}
```

- [ ] **Step 4: Run to verify pass**

Run: `bun test test/session/moa.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/opencode/src/session/moa.ts packages/opencode/test/session/moa.test.ts
git commit -m "feat(moa): add reference view, tool-list, and context injection"
```

---

### Task 3: Add `moa` block to the config schema

**Files:**
- Modify: `packages/opencode/src/config/config.ts` (add to `Info` zod object, near `small_model` ~line 140)
- Test: `packages/opencode/test/session/moa.test.ts` (add a schema-parse test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Config.Info` now carries an optional `moa` field of shape:
  `{ default_preset?: string; reference_concurrency?: number; presets: Record<string, { enabled?: boolean; reference_models: string[]; aggregator: string; max_tokens?: number }> }`.
  Downstream (Tasks 4–5) read `cfg.moa` and pass it through `normalizeMoAConfig` from Task 1.

- [ ] **Step 1: Write the failing test**

Append to `packages/opencode/test/session/moa.test.ts`:

```ts
import { Config } from "../../src/config"

describe("Config.Info moa schema", () => {
  test("parses a moa block", () => {
    const parsed = Config.Info.parse({
      moa: {
        default_preset: "default",
        reference_concurrency: 4,
        presets: {
          default: {
            reference_models: ["openai/gpt-5.5", "openrouter/deepseek/deepseek-v4-pro"],
            aggregator: "anthropic/claude-opus-4.8",
            max_tokens: 2048,
          },
        },
      },
    })
    expect(parsed.moa?.presets.default.aggregator).toBe("anthropic/claude-opus-4.8")
    expect(parsed.moa?.reference_concurrency).toBe(4)
  })

  test("moa is optional", () => {
    expect(() => Config.Info.parse({})).not.toThrow()
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `bun test test/session/moa.test.ts`
Expected: FAIL — `moa` stripped/invalid (depending on zod mode) or the assertion on `parsed.moa` fails.

- [ ] **Step 3: Add the schema**

In `packages/opencode/src/config/config.ts`, immediately after the `small_model` field (line ~140), add:

```ts
    moa: z
      .object({
        default_preset: z.string().optional().describe("Default MoA preset name"),
        reference_concurrency: z
          .number()
          .int()
          .positive()
          .optional()
          .describe("Max reference models to call in parallel (default 8)"),
        presets: z.record(
          z.string(),
          z.object({
            enabled: z.boolean().optional().describe("Enable reference fan-out; false ⇒ aggregator acts alone"),
            reference_models: z
              .array(ConfigModelID)
              .min(1)
              .describe("Advisory models run in parallel, in provider/model form"),
            aggregator: ConfigModelID.describe("The acting model that calls tools, in provider/model form"),
            max_tokens: z.number().int().positive().optional().describe("Output cap; unset ⇒ model maximum"),
          }),
        ),
      })
      .optional()
      .describe("Mixture of Agents presets, selectable as models under the 'moa' provider"),
```

- [ ] **Step 4: Run to verify pass**

Run: `bun test test/session/moa.test.ts`
Expected: PASS.

- [ ] **Step 5: Run the broader config suite (no regressions)**

Run: `bun test test/config/config.test.ts`
Expected: PASS (unchanged behavior).

- [ ] **Step 6: Commit**

```bash
git add packages/opencode/src/config/config.ts packages/opencode/test/session/moa.test.ts
git commit -m "feat(moa): add moa presets to config schema"
```

---

### Task 4: Synthesize the `moa` virtual provider

**Files:**
- Modify: `packages/opencode/src/provider/provider.ts` (insert before `return { models: languages, providers, ... }` at ~line 1525; also guard `getLanguage` at ~line 1719)
- Test: `packages/opencode/test/provider/moa-provider.test.ts`

**Interfaces:**
- Consumes: `normalizeMoAConfig` (Task 1); `cfg` (from `config.get()`, already bound at line 1213); the in-scope `providers: Record<string, Info>` map; `Model`/`Info` shapes.
- Produces: a `providers["moa"]` entry (when `cfg.moa` has presets) whose `models[<preset>]` is a `Model` with `providerID: "moa"`, `id: <preset>`, `name: "MoA: <preset>"`, and capability/limit/cost fields COPIED from the resolved aggregator's `Model`. `getLanguage` throws for `moa/*`.

- [ ] **Step 1: Write the failing test**

Create `packages/opencode/test/provider/moa-provider.test.ts` (follow the imports/`run` helper pattern from `packages/opencode/test/provider/provider.test.ts:1-30`):

```ts
import { test, expect } from "bun:test"
import { Effect } from "effect"
import { Provider } from "../../src/provider"
import { Config } from "../../src/config"
import { ProviderID, ModelID } from "../../src/provider/schema"
import { AppRuntime } from "../../src/effect/app-runtime"

async function run<A, E>(fn: (p: Provider.Interface) => Effect.Effect<A, E, never>) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const provider = yield* Provider.Service
      return yield* fn(provider)
    }),
  )
}

test("moa provider exposes presets as models with MoA: prefix and aggregator capabilities", async () => {
  // Assumes test config provides an `anthropic` provider with a tool-calling model
  // and a moa preset whose aggregator points at it. See fixture setup in this file.
  const list = await run((p) => p.list())
  const moa = list.all.find((x) => x.id === "moa")
  expect(moa).toBeDefined()
  const model = moa!.models["default"]
  expect(model.name).toBe("MoA: default")
  expect(model.capabilities.tool_call).toBe(true) // inherited from aggregator
})

test("getLanguage on a moa model throws (not directly callable)", async () => {
  const err = await run((p) =>
    p
      .getModel(ProviderID.make("moa"), ModelID.make("default"))
      .pipe(Effect.flatMap((m) => p.getLanguage(m)), Effect.flip),
  )
  expect(String(err)).toMatch(/moa.*aggregator|not directly callable/i)
})
```

Note: the fixture must write a config with an `anthropic` provider (or another local test provider that has a tool-calling model) plus `moa.presets.default.aggregator` pointing to it. Model the fixture on how `provider.test.ts` seeds config via `tmpdir` + `clearConfig`; if that harness is unavailable, seed `cfg.moa` through the same `Config` layer the other provider tests use.

- [ ] **Step 2: Add the synthesized-model helper to `moa.ts`**

Append to `packages/opencode/src/session/moa.ts`:

```ts
// Build a synthesized MoA preset Model by copying capability-bearing fields
// from the aggregator's Model. `Model` is intentionally typed loosely here to
// avoid a provider→session import cycle; the provider layer passes its real
// Model and gets a Model-shaped object back.
export function synthesizePresetModel(presetName: string, aggregatorModel: any): any {
  return {
    ...aggregatorModel,
    id: presetName,
    providerID: "moa",
    name: `MoA: ${presetName}`,
  }
}
```

- [ ] **Step 3: Run the helper's unit test**

Append to `packages/opencode/test/session/moa.test.ts`:

```ts
import { synthesizePresetModel } from "../../src/session/moa"

describe("synthesizePresetModel", () => {
  test("copies aggregator capabilities, overrides id/provider/name", () => {
    const agg = { id: "claude-opus-4.8", providerID: "anthropic", capabilities: { tool_call: true }, limit: { context: 200000, output: 8192 } }
    const m = synthesizePresetModel("default", agg)
    expect(m.id).toBe("default")
    expect(m.providerID).toBe("moa")
    expect(m.name).toBe("MoA: default")
    expect(m.capabilities.tool_call).toBe(true)
    expect(m.limit.context).toBe(200000)
  })
})
```

Run: `bun test test/session/moa.test.ts`
Expected: PASS.

- [ ] **Step 4: Wire synthesis into provider assembly**

In `packages/opencode/src/provider/provider.ts`, add the import near the top with the other `@/session`-free imports:

```ts
import { normalizeMoAConfig, synthesizePresetModel, parseSlot } from "@/session/moa"
```

Then, immediately BEFORE `return { models: languages, providers, sdk, modelLoaders, varsLoaders }` (~line 1525), insert:

```ts
        // Synthesize the virtual `moa` provider: one model per configured preset.
        const moaCfg = normalizeMoAConfig((cfg as any).moa)
        if (Object.keys(moaCfg.presets).length > 0) {
          const moaModels: Record<string, any> = {}
          for (const [presetName, preset] of Object.entries(moaCfg.presets)) {
            const agg = preset.aggregator
            const aggProvider = providers[agg.providerID]
            const aggModel = aggProvider?.models?.[agg.modelID]
            if (!aggModel) {
              log.warn("moa preset aggregator not found; skipping preset", {
                preset: presetName,
                aggregator: `${agg.providerID}/${agg.modelID}`,
              })
              continue
            }
            moaModels[presetName] = synthesizePresetModel(presetName, aggModel)
          }
          if (Object.keys(moaModels).length > 0) {
            providers[ProviderID.make("moa")] = {
              id: ProviderID.make("moa"),
              name: "Mixture of Agents",
              source: "config",
              env: [],
              options: {},
              models: moaModels,
            } as any
          }
        }
```

- [ ] **Step 5: Guard `getLanguage` against `moa/*`**

In `getLanguage` (`provider.ts:1719`), add at the very start of the function body (before the cache lookup):

```ts
      if (model.providerID === "moa")
        throw new Error(
          `moa provider models are resolved via their aggregator; "${model.id}" is not directly callable`,
        )
```

- [ ] **Step 6: Run the provider tests**

Run: `bun test test/provider/moa-provider.test.ts`
Expected: PASS.

- [ ] **Step 7: Run the full provider suite (no regressions)**

Run: `bun test test/provider/`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add packages/opencode/src/provider/provider.ts packages/opencode/src/session/moa.ts \
  packages/opencode/test/provider/moa-provider.test.ts packages/opencode/test/session/moa.test.ts
git commit -m "feat(moa): register moa as a virtual provider with aggregator capabilities"
```

---

### Task 5: Reference fan-out helper with concurrency + failure degradation (pure)

**Files:**
- Modify: `packages/opencode/src/session/moa.ts`
- Test: `packages/opencode/test/session/moa.test.ts`

**Interfaces:**
- Consumes: nothing new (takes injected async callbacks, so it is testable without the AI SDK).
- Produces:
  - `type ReferenceCall = { label: string; call: () => Promise<string> }`
  - `function runReferenceFanout(calls: ReferenceCall[], concurrency: number): Promise<{ label: string; text: string }[]>` — runs calls with a bounded pool (max `concurrency` in flight), preserves input order in the output, and turns a thrown/rejected call into `{ label, text: "[failed: <msg>]" }` instead of rejecting the whole batch.

- [ ] **Step 1: Write failing tests**

Append to `packages/opencode/test/session/moa.test.ts`:

```ts
import { runReferenceFanout, type ReferenceCall } from "../../src/session/moa"

describe("runReferenceFanout", () => {
  test("returns outputs in input order", async () => {
    const calls: ReferenceCall[] = [
      { label: "a", call: async () => "AA" },
      { label: "b", call: async () => "BB" },
    ]
    expect(await runReferenceFanout(calls, 8)).toEqual([
      { label: "a", text: "AA" },
      { label: "b", text: "BB" },
    ])
  })

  test("a failing call degrades to a [failed: …] note without aborting others", async () => {
    const calls: ReferenceCall[] = [
      { label: "ok", call: async () => "fine" },
      { label: "bad", call: async () => { throw new Error("boom") } },
    ]
    const out = await runReferenceFanout(calls, 8)
    expect(out[0]).toEqual({ label: "ok", text: "fine" })
    expect(out[1].label).toBe("bad")
    expect(out[1].text).toMatch(/\[failed: .*boom.*\]/)
  })

  test("respects the concurrency bound", async () => {
    let inFlight = 0
    let maxInFlight = 0
    const make = (v: string): ReferenceCall => ({
      label: v,
      call: async () => {
        inFlight++
        maxInFlight = Math.max(maxInFlight, inFlight)
        await new Promise((r) => setTimeout(r, 5))
        inFlight--
        return v
      },
    })
    await runReferenceFanout([make("1"), make("2"), make("3"), make("4")], 2)
    expect(maxInFlight).toBeLessThanOrEqual(2)
  })

  test("empty input ⇒ empty output", async () => {
    expect(await runReferenceFanout([], 8)).toEqual([])
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `bun test test/session/moa.test.ts`
Expected: FAIL — `runReferenceFanout` not exported.

- [ ] **Step 3: Implement in `moa.ts`**

Append to `packages/opencode/src/session/moa.ts`:

```ts
export type ReferenceCall = { label: string; call: () => Promise<string> }

export async function runReferenceFanout(
  calls: ReferenceCall[],
  concurrency: number,
): Promise<{ label: string; text: string }[]> {
  const results: { label: string; text: string }[] = new Array(calls.length)
  let next = 0
  const limit = Math.max(1, concurrency)
  async function worker() {
    while (true) {
      const i = next++
      if (i >= calls.length) return
      const { label, call } = calls[i]
      try {
        results[i] = { label, text: await call() }
      } catch (e) {
        results[i] = { label, text: `[failed: ${e instanceof Error ? e.message : String(e)}]` }
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, calls.length) }, worker))
  return results
}
```

- [ ] **Step 4: Run to verify pass**

Run: `bun test test/session/moa.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/opencode/src/session/moa.ts packages/opencode/test/session/moa.test.ts
git commit -m "feat(moa): add bounded reference fan-out with failure degradation"
```

---

### Task 6: Wire MoA into `llm.ts` — aggregator swap in `prepare`, fan-out in `run`

**Files:**
- Modify: `packages/opencode/src/session/llm.ts`
- Test: covered by Task 4's provider tests + Task 7's E2E; this task adds no new unit test (it is thin wiring over already-tested pure helpers), but MUST pass typecheck and the existing `llm` suite.

**Interfaces:**
- Consumes: `normalizeMoAConfig`, `resolveMoAPreset`, `parseSlot`, `referenceMessages`, `toolListText`, `referenceSystemPrompt`, `synthesizeContext`, `injectContext`, `runReferenceFanout`, `type ReferenceCall` (all from `@/session/moa`, Tasks 1–5); `generateText` from `ai`; `provider.getModel` / `provider.getLanguage`.
- Produces: MoA behavior at runtime. `PreparedInput` gains an optional `moa` field carrying the resolved reference plan so `run()` can fan out.

**Design rationale (read before editing):** `prepare()` calls `getLanguage(input.model)` at its top, which the Task 4 guard makes throw for `moa/*`. Therefore, at the very start of `prepare()`, detect a `moa/*` model, resolve the preset, and REPLACE `input.model` with the aggregator's `Model` (the aggregator IS the acting model, so the system prompt, capabilities, and `getLanguage` all correctly use it). Stash the reference plan on the returned `PreparedInput`. In `run()`, after `tools` and `messages` are built, if a plan is stashed and enabled, fan out the references and inject the context into `messages` before `streamText`.

- [ ] **Step 1: Add imports**

In `packages/opencode/src/session/llm.ts`, extend the `ai` import (line 5) to include `generateText`:

```ts
import { streamText, generateText, wrapLanguageModel, type ModelMessage, type Tool, tool, jsonSchema } from "ai"
```

Add the MoA module import near the other `@/` imports:

```ts
import {
  normalizeMoAConfig,
  resolveMoAPreset,
  referenceMessages,
  toolListText,
  referenceSystemPrompt,
  synthesizeContext,
  injectContext,
  runReferenceFanout,
  type ReferenceCall,
  type MoASlot,
} from "@/session/moa"
```

- [ ] **Step 2: Extend the `PreparedInput` type**

In the `PreparedInput` type (line ~54), add an optional `moa` field:

```ts
export type PreparedInput = StreamInput & {
  abort?: AbortSignal
  telemetry: RequestTelemetry
  moa?: {
    presetName: string
    referenceModels: MoASlot[]
    aggregatorLabel: string
    concurrency: number
    maxTokens?: number
  }
  request: {
    system: string[]
    messages: ModelMessage[]
    isOpenaiOauth: boolean
  }
}
```

- [ ] **Step 3: Swap MoA model → aggregator at the top of `prepare()`**

In `prepare()`, immediately after `if (isPreparedInput(input)) return input`, insert:

```ts
      let moaPlan: PreparedInput["moa"] | undefined
      if (input.model.providerID === "moa") {
        const cfg = yield* config.get()
        const moaCfg = normalizeMoAConfig((cfg as any).moa)
        const preset = resolveMoAPreset(moaCfg, input.model.id) // throws if unknown → aborts turn
        const agg = preset.aggregator
        const aggregatorModel = yield* provider.getModel(
          ProviderID.make(agg.providerID),
          ModelID.make(agg.modelID),
        ) // throws if aggregator unresolvable → aborts turn (intended: no acting model)
        if (preset.enabled && preset.reference_models.length > 0) {
          moaPlan = {
            presetName: input.model.id,
            referenceModels: preset.reference_models,
            aggregatorLabel: `${agg.providerID}/${agg.modelID}`,
            concurrency: moaCfg.reference_concurrency,
            maxTokens: preset.max_tokens,
          }
        }
        // The aggregator is the acting model: everything downstream uses it.
        input = { ...input, model: aggregatorModel }
      }
```

Then, in the object `prepare()` returns (the `satisfies PreparedInput` literal), add the `moa` field:

```ts
      return {
        ...input,
        moa: moaPlan,
        telemetry: {
```

Add the needed schema imports at the top of `llm.ts` if not present:

```ts
import { ProviderID, ModelID } from "@/provider/schema"
```

- [ ] **Step 4: Run references + inject in `run()` before `streamText`**

In `run()`, find `const tools = resolveTools(prepared)` (line ~230). AFTER it and the LiteLLM stub block, but BEFORE `const tracer = ...` (line ~351), insert:

```ts
      let messages = prepared.request.messages
      if (prepared.moa) {
        const plan = prepared.moa
        const refView = referenceMessages(messages)
        const toolList = toolListText(
          Object.entries(tools).map(([name, t]) => ({
            name,
            description: (t as Tool).description as string | undefined,
          })),
        )
        const refSystem = referenceSystemPrompt(toolList)
        const calls: ReferenceCall[] = []
        for (const slot of plan.referenceModels) {
          const refModel = yield* provider.getModel(
            ProviderID.make(slot.providerID),
            ModelID.make(slot.modelID),
          )
          const refLanguage = yield* provider.getLanguage(refModel)
          const label = `${slot.providerID}/${slot.modelID}`
          calls.push({
            label,
            call: async () => {
              const res = await generateText({
                model: refLanguage,
                system: refSystem,
                messages: refView,
                ...(plan.maxTokens ? { maxOutputTokens: plan.maxTokens } : {}),
                abortSignal: prepared.abort,
              })
              return res.text || "(empty response)"
            },
          })
        }
        const outputs = yield* Effect.promise(() => runReferenceFanout(calls, plan.concurrency))
        const context = synthesizeContext({
          preset: plan.presetName,
          aggregatorLabel: plan.aggregatorLabel,
          referenceLabels: calls.map((c) => c.label),
          outputs,
        })
        messages = injectContext(messages, context)
      }
```

Then, in the `streamText({...})` call, change `messages,` (line ~409) to use the possibly-injected local:

```ts
          messages,
```

is already the local `messages` — since the original code had `const messages = prepared.request.messages` at line 194. **Remove that original `const messages = prepared.request.messages` at line ~194** (the new `let messages = ...` block in this step replaces it). Verify there is exactly one `messages` binding after the edit.

- [ ] **Step 5: Typecheck**

Run (from `packages/opencode/`): `bunx tsc --noEmit -p tsconfig.json` (or the repo's typecheck script if defined in `package.json`).
Expected: no type errors in `llm.ts` / `moa.ts` / `provider.ts`.

- [ ] **Step 6: Run the existing llm + session suites (no regressions)**

Run: `bun test test/session/llm.test.ts test/session/`
Expected: PASS (MoA path is inert when no `moa/*` model is selected).

- [ ] **Step 6b: Verify the `enabled: false` degradation path**

The disabled path is: preset resolves, `input.model` is swapped to the aggregator, but `moaPlan` stays `undefined` (because `preset.enabled` is false), so `run()` skips the fan-out entirely and streams the aggregator alone. Confirm this by reading the Step 3 condition (`if (preset.enabled && preset.reference_models.length > 0)`) — no reference calls are dispatched when disabled. Add a provider-level assertion in `packages/opencode/test/provider/moa-provider.test.ts` that a `moa/default` preset with `enabled: false` still resolves to a model whose capabilities match the aggregator (proving the swap path works without fan-out):

```ts
test("disabled preset still exposes an aggregator-backed model", async () => {
  // fixture: moa.presets.default.enabled = false
  const list = await run((p) => p.list())
  const model = list.all.find((x) => x.id === "moa")?.models["default"]
  expect(model?.capabilities.tool_call).toBe(true)
})
```

- [ ] **Step 7: Commit**

```bash
git add packages/opencode/src/session/llm.ts
git commit -m "feat(moa): run reference fan-out and inject context in the llm loop"
```

---

### Task 7: Real multi-model end-to-end test (env-gated)

**Files:**
- Create: `packages/opencode/test/session/moa-integration.test.ts`

**Interfaces:**
- Consumes: the full stack (config → provider → llm). Requires real credentials for the configured providers.

**Why env-gated:** this makes real network calls and needs real API keys, so it must not run in the default CI suite. It runs only when `MOA_E2E=1` is set, mirroring how hermes keeps gateway-level tests out of the default run.

- [ ] **Step 1: Write the gated E2E test**

Create `packages/opencode/test/session/moa-integration.test.ts`:

```ts
import { test, expect } from "bun:test"

const RUN = process.env.MOA_E2E === "1"
const maybe = RUN ? test : test.skip

// Requires a real config on disk (or via the same Config layer other tests use)
// with credentials for the providers named below, e.g.:
//   moa:
//     presets:
//       default:
//         reference_models: ["<provider>/<ref-model-1>", "<provider>/<ref-model-2>"]
//         aggregator: "<provider>/<tool-calling-model>"
//
// Set the two reference models and the aggregator via env so this file has no
// hardcoded model ids:
//   MOA_E2E=1 MOA_AGG="anthropic/claude-opus-4.8" \
//   MOA_REF1="openai/gpt-5.5" MOA_REF2="openrouter/deepseek/deepseek-v4-pro" \
//   bun test test/session/moa-integration.test.ts

maybe("MoA runs references in parallel and the aggregator acts with tools", async () => {
  // Import lazily so the module graph (and its network-touching deps) only
  // loads under MOA_E2E=1.
  const { Effect } = await import("effect")
  const { Provider } = await import("../../src/provider")
  const { LLM } = await import("../../src/session/llm")
  const { AppRuntime } = await import("../../src/effect/app-runtime")
  const { ProviderID, ModelID } = await import("../../src/provider/schema")

  const text = await AppRuntime.runPromise(
    Effect.gen(function* () {
      const provider = yield* Provider.Service
      const model = yield* provider.getModel(ProviderID.make("moa"), ModelID.make("default"))
      const llm = yield* LLM.Service
      // Build a minimal StreamInput; reuse the same construction the title-gen
      // path uses in prompt.ts:199 as a template (agent, user, system, tools).
      // Collect the streamed text-delta output.
      const parts: string[] = []
      yield* llm
        .stream({
          agent: /* a minimal build agent — see prompt.ts for shape */ (yield* makeMinimalAgent()),
          user: yield* makeMinimalUser("List the files in this directory, then say DONE."),
          system: [],
          tools: yield* buildRealTools(), // include a read-only tool like `list`/`bash`
          model,
          sessionID: yield* makeSession(),
          messages: [{ role: "user", content: "List the files in this directory, then say DONE." }],
        } as any)
        .pipe(
          Stream.filter((e: any): e is { type: "text-delta"; text: string } => e.type === "text-delta"),
          Stream.tap((e: any) => Effect.sync(() => parts.push(e.text))),
          Stream.runDrain,
        )
      return parts.join("")
    }),
  )
  expect(text.length).toBeGreaterThan(0)
})
```

Note: the helper stubs (`makeMinimalAgent`, `makeMinimalUser`, `buildRealTools`, `makeSession`, and the `Stream` import) must be filled in by mirroring an existing end-to-end session test in `packages/opencode/test/session/` (e.g. `prompt.test.ts` / `processor-effect.test.ts`) — copy their setup verbatim rather than inventing new scaffolding. If a full stream harness proves too heavy, the acceptable minimum for this task is: select `moa/default`, drive one real turn through `llm.stream`, and assert non-empty streamed output plus (from logs or a spy) that ≥2 reference calls were dispatched.

- [ ] **Step 2: Verify it SKIPS by default**

Run: `bun test test/session/moa-integration.test.ts`
Expected: the test is reported as skipped (no network, no credentials needed).

- [ ] **Step 3: Verify it RUNS with real credentials (manual)**

Run (with real keys configured):
`MOA_E2E=1 MOA_AGG=... MOA_REF1=... MOA_REF2=... bun test test/session/moa-integration.test.ts`
Expected: PASS — non-empty streamed output; references fired in parallel; aggregator acted (optionally executed a read-only tool).

- [ ] **Step 4: Commit**

```bash
git add packages/opencode/test/session/moa-integration.test.ts
git commit -m "test(moa): add env-gated real multi-model e2e"
```

---

## Final verification

- [ ] Run the full opencode suite: `bun test` (from `packages/opencode/`). Expected: PASS (the E2E test skips).
- [ ] Typecheck the package. Expected: clean.
- [ ] Manually confirm selection UX: with a `moa.presets.default` configured, `moa/default` appears in the model list as `MoA: default` and `/model default --provider moa` selects it.

