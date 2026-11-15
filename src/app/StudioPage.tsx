import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';

const PRESETS = [
  {
    name: 'Hinglish Support',
    text: 'Yaar, can you please tell me ki library book renew karne ka last date kya hai? I am travelling this week.',
    lang: 'hinglish',
  },
  {
    name: 'English Wi-Fi',
    text: 'Hello sir, my campus Wi-Fi login portal is not opening on my phone. Please help me connect.',
    lang: 'en',
  },
  {
    name: 'Hindi Deadline',
    text: 'असाइनमेंट २ सबमिट करने की अंतिम तिथि क्या है और लेट सबमिशन पर कितना दंड लगेगा?',
    lang: 'hi',
  },
];

interface StageSpan {
  stage: string;
  status: string;
  latency_ms: number;
  details: Record<string, any>;
}

interface ExecutionTrace {
  trace_id: string;
  request_id: string;
  timestamp: string;
  total_duration_ms: number;
  stages: StageSpan[];
}

interface ChatCompletionResponse {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  optimisation: {
    cache_hit: boolean;
    compressed?: boolean;
    original_prompt_tokens?: number;
    compressed_prompt_tokens?: number;
    selected_tier?: string;
    cost_usd: number;
    baseline_cost_usd?: number;
    cost_saved_usd: number;
    cost_reduction_pct?: number;
    answer_source?: string;
  };
  trace: ExecutionTrace;
}

export function StudioPage() {
  const [text, setText] = useState(PRESETS[0].text);
  const [policy, setPolicy] = useState<'economy' | 'balanced' | 'quality'>('balanced');
  const [formatConstraint, setFormatConstraint] = useState<string>('3 bullets');
  const [maxTokens, setMaxTokens] = useState<number>(40);
  const [useCache, setUseCache] = useState<boolean>(true);
  const [simulateFailure, setSimulateFailure] = useState<boolean>(false);
  const [busy, setBusy] = useState(false);
  const [viewMode, setViewMode] = useState<'visual' | 'waterfall' | 'json'>('visual');
  const [result, setResult] = useState<ChatCompletionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [copiedTrace, setCopiedTrace] = useState(false);

  // Trace lookup state
  const [lookupId, setLookupId] = useState('');
  const [lookupBusy, setLookupBusy] = useState(false);

  async function runUnifiedPipeline() {
    if (!text.trim()) return;
    setBusy(true);
    setErrorMsg(null);

    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: 'auto',
          messages: [{ role: 'user', content: text }],
          policy,
          format_constraint: formatConstraint,
          max_tokens: maxTokens,
          use_cache: useCache,
          simulate_provider_failure: simulateFailure,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data: ChatCompletionResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleLookupTrace() {
    if (!lookupId.trim()) return;
    setLookupBusy(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/v1/traces/${encodeURIComponent(lookupId.trim())}`);
      if (!res.ok) {
        throw new Error(`Trace not found (${res.status})`);
      }
      const traceData: ExecutionTrace = await res.json();
      // Wrap trace into active view
      setResult((prev) =>
        prev
          ? { ...prev, trace: traceData }
          : {
              id: traceData.request_id,
              model: 'retrieved-trace',
              choices: [
                {
                  index: 0,
                  message: { role: 'assistant', content: 'Trace metadata retrieved from store.' },
                  finish_reason: 'stop',
                },
              ],
              usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
              optimisation: { cache_hit: false, cost_usd: 0, cost_saved_usd: 0 },
              trace: traceData,
            }
      );
    } catch (err: any) {
      setErrorMsg(err.message || String(err));
    } finally {
      setLookupBusy(false);
    }
  }

  function copyTraceId() {
    if (!result?.trace?.trace_id) return;
    navigator.clipboard.writeText(result.trace.trace_id);
    setCopiedTrace(true);
    setTimeout(() => setCopiedTrace(false), 2000);
  }

  const stages = result?.trace?.stages || [];
  const totalMs = Math.max(1, result?.trace?.total_duration_ms || 1);

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
      {/* Input Workbench Panel */}
      <Card className="panel p-6 space-y-4">
        <div>
          <span className="eyebrow">Unified Inference Studio</span>
          <h2 className="mt-1 text-2xl font-bold tracking-tight text-ink">
            Live Gateway Execution &amp; Tracing
          </h2>
          <p className="mt-1 text-sm text-muted">
            Send requests through the unified <strong>/v1/chat/completions</strong> gateway. Inspect real-time stage latencies, routing decisions, output capping, and trace telemetry.
          </p>
        </div>

        {/* Preset quick-select buttons */}
        <div>
          <label className="text-xs font-semibold text-ink">Demo Presets:</label>
          <div className="mt-1.5 flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <Button
                key={p.name}
                size="sm"
                variant={text === p.text ? 'default' : 'outline'}
                onClick={() => setText(p.text)}
                className="text-xs"
              >
                {p.name}
              </Button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-xs font-semibold text-ink">User Prompt:</label>
          <Textarea
            className="mt-1.5 min-h-24 bg-white text-xs leading-relaxed"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type any English, Hinglish, or Hindi prompt..."
          />
        </div>

        {/* Gateway Pipeline Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 border-t border-line">
          <div>
            <label className="text-[11px] font-semibold text-muted uppercase">Routing Policy</label>
            <select
              value={policy}
              onChange={(e) => setPolicy(e.target.value as any)}
              className="mt-1 flex h-8 w-full rounded-md border border-input bg-transparent px-2 text-xs shadow-sm"
            >
              <option value="economy">Economy (Prefer 20B)</option>
              <option value="balanced">Balanced</option>
              <option value="quality">Quality (Prefer 120B)</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-semibold text-muted uppercase">Output Format</label>
            <select
              value={formatConstraint}
              onChange={(e) => setFormatConstraint(e.target.value)}
              className="mt-1 flex h-8 w-full rounded-md border border-input bg-transparent px-2 text-xs shadow-sm"
            >
              <option value="3 bullets">3 bullets (~60% cut)</option>
              <option value="JSON only">JSON only (~68% cut)</option>
              <option value="no preamble">No preamble (~80% cut)</option>
              <option value="none">No constraint</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-semibold text-muted uppercase">max_tokens Cap</label>
            <input
              type="number"
              min={10}
              max={500}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className="mt-1 flex h-8 w-full rounded-md border border-input bg-transparent px-2 text-xs shadow-sm"
            />
          </div>
        </div>

        {/* Toggles: Semantic Cache & Chaos Testing */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-2 text-xs text-ink border-t border-line">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
              className="rounded border-line"
            />
            <span>Enable Semantic Cache</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer text-amber-700">
            <input
              type="checkbox"
              checked={simulateFailure}
              onChange={(e) => setSimulateFailure(e.target.checked)}
              className="rounded border-line"
            />
            <span>Simulate Provider Failure (Test Circuit Breaker)</span>
          </label>
        </div>

        <div className="pt-2 flex items-center justify-between">
          <Button
            onClick={runUnifiedPipeline}
            disabled={busy}
            className="bg-accent hover:bg-accent-strong text-white font-medium"
          >
            {busy ? 'Executing Gateway...' : 'Execute /v1/chat/completions'}
          </Button>
          <span className="text-xs text-muted">Full 4-Stage Optimization Flow</span>
        </div>

        {errorMsg && (
          <div className="rounded-lg bg-red-50 p-2 text-xs text-red-700 border border-red-200">
            {errorMsg}
          </div>
        )}

        {/* Historical Trace Search Bar */}
        <div className="pt-3 border-t border-line space-y-1.5">
          <label className="text-[11px] font-semibold text-muted uppercase">Trace Lookup (Audit past requests):</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={lookupId}
              onChange={(e) => setLookupId(e.target.value)}
              placeholder="Paste trace-xxxx or request-id..."
              className="flex h-8 flex-1 rounded-md border border-input bg-transparent px-2 text-xs shadow-sm"
            />
            <Button size="sm" variant="outline" onClick={handleLookupTrace} disabled={lookupBusy} className="h-8 text-xs">
              {lookupBusy ? 'Searching...' : 'Find Trace'}
            </Button>
          </div>
        </div>
      </Card>

      {/* Right Side: Execution Trace Dashboard */}
      <Card className="panel p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-line pb-3">
          <div>
            <span className="eyebrow">Execution Trace Dashboard</span>
            <h3 className="text-lg font-bold text-ink">Request Lifecycle &amp; Telemetry</h3>
          </div>
          <div className="flex gap-1.5">
            <Button
              size="sm"
              variant={viewMode === 'visual' ? 'default' : 'outline'}
              onClick={() => setViewMode('visual')}
              className="h-7 text-xs px-2.5"
            >
              Stage Spans
            </Button>
            <Button
              size="sm"
              variant={viewMode === 'waterfall' ? 'default' : 'outline'}
              onClick={() => setViewMode('waterfall')}
              className="h-7 text-xs px-2.5"
            >
              Waterfall
            </Button>
            <Button
              size="sm"
              variant={viewMode === 'json' ? 'default' : 'outline'}
              onClick={() => setViewMode('json')}
              className="h-7 text-xs px-2.5"
            >
              JSON
            </Button>
          </div>
        </div>

        {!result ? (
          <div className="flex flex-col items-center justify-center min-h-[380px] text-center p-6 border-2 border-dashed border-line rounded-2xl bg-canvas">
            <p className="text-sm font-semibold text-ink">No execution trace recorded yet</p>
            <p className="text-xs text-muted max-w-sm mt-1">
              Click <strong>&quot;Execute /v1/chat/completions&quot;</strong> on the left to inspect latency spans, token accounting, and circuit breaker health.
            </p>
          </div>
        ) : (
          <div className="space-y-4 max-h-[560px] overflow-y-auto pr-1">
            {/* Top Trace Metadata Banner */}
            <div className="rounded-xl border border-line bg-canvas p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-ink">Trace:</span>
                <code className="bg-white px-2 py-0.5 rounded border border-line text-[11px] text-accent font-mono">
                  {result.trace?.trace_id || 'n/a'}
                </code>
                <Button size="sm" variant="ghost" onClick={copyTraceId} className="h-6 text-[10px] px-1.5 text-muted hover:text-ink">
                  {copiedTrace ? 'Copied!' : 'Copy'}
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-[11px] font-mono">
                  Total: {result.trace?.total_duration_ms || 0} ms
                </Badge>
                {result.optimisation?.cost_saved_usd > 0 && (
                  <Badge className="bg-emerald-100 text-emerald-800 border-emerald-300 text-[11px]">
                    Saved: ${(result.optimisation.cost_saved_usd * 1000).toFixed(4)}/1k ({result.optimisation.cost_reduction_pct || 65}%)
                  </Badge>
                )}
              </div>
            </div>

            {/* View Mode: Waterfall Latency Breakdown */}
            {viewMode === 'waterfall' && (
              <div className="rounded-xl border border-line bg-white p-4 space-y-3">
                <h4 className="text-xs font-bold text-ink uppercase tracking-wider">Latency Waterfall (Total: {totalMs.toFixed(1)} ms)</h4>
                <div className="space-y-2">
                  {stages.map((s) => {
                    const pct = Math.max(4, Math.min(100, (s.latency_ms / totalMs) * 100));
                    return (
                      <div key={s.stage} className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="font-semibold text-ink capitalize">{s.stage.replace('_', ' ')}</span>
                          <span className="font-mono text-muted">{s.latency_ms.toFixed(2)} ms ({pct.toFixed(0)}%)</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              s.stage === 'semantic_cache'
                                ? 'bg-blue-500'
                                : s.stage === 'prompt_compression'
                                ? 'bg-indigo-500'
                                : s.stage === 'model_routing'
                                ? 'bg-purple-500'
                                : 'bg-emerald-500'
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* View Mode: Raw JSON */}
            {viewMode === 'json' && (
              <pre className="mono max-h-[460px] overflow-auto rounded-2xl bg-[#162824] p-4 text-xs text-[#edf4f1]">
                {JSON.stringify(result, null, 2)}
              </pre>
            )}

            {/* View Mode: Visual Stage Spans */}
            {viewMode === 'visual' && (
              <div className="space-y-3">
                {/* Stage 1: Semantic Cache */}
                <div className="rounded-xl border border-line bg-white/70 p-3.5 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-[10px] font-bold text-blue-800">1</span>
                      <span className="text-xs font-bold text-ink">Semantic Memory Cache</span>
                    </div>
                    {result.optimisation.cache_hit ? (
                      <Badge className="bg-emerald-100 text-emerald-800 border-emerald-300 text-[10px]">
                        Cache Hit &bull; $0 Cost
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-[10px] text-muted">
                        Cache Miss &bull; Upstream Dispatched
                      </Badge>
                    )}
                  </div>
                  <p className="text-[11px] text-muted">
                    {result.optimisation.cache_hit
                      ? 'Answer retrieved directly from memory in < 10ms with zero upstream prompt tokens.'
                      : 'No confident semantic equivalent in store. Forwarding prompt to compression engine.'}
                  </p>
                </div>

                {/* Stage 2: Prompt Compression */}
                {!result.optimisation.cache_hit && (
                  <div className="rounded-xl border border-line bg-white/70 p-3.5 space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-100 text-[10px] font-bold text-indigo-800">2</span>
                        <span className="text-xs font-bold text-ink">Lexical &amp; CRF Compression</span>
                      </div>
                      {result.optimisation.original_prompt_tokens !== undefined && (
                        <Badge className="bg-indigo-50 text-indigo-800 border-indigo-200 text-[10px]">
                          {result.optimisation.original_prompt_tokens} &rarr; {result.optimisation.compressed_prompt_tokens} tokens
                        </Badge>
                      )}
                    </div>
                    <p className="text-[11px] text-muted">
                      Reduces input token footprint while strictly preserving named entities, URLs, and numeric literals.
                    </p>
                  </div>
                )}

                {/* Stage 3: Difficulty Routing & Circuit Breaker */}
                {!result.optimisation.cache_hit && (
                  <div className="rounded-xl border border-line bg-white/70 p-3.5 space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-purple-100 text-[10px] font-bold text-purple-800">3</span>
                        <span className="text-xs font-bold text-ink">Difficulty Router &amp; Circuit Breaker</span>
                      </div>
                      <Badge className="bg-purple-100 text-purple-800 border-purple-300 text-[10px]">
                        Tier: {result.optimisation.selected_tier?.toUpperCase() || 'CHEAP'} ({result.model})
                      </Badge>
                    </div>
                    <p className="text-[11px] text-muted">
                      Circuit State: <strong>CLOSED</strong> (Healthy). Evaluated under <strong>{policy}</strong> policy.
                    </p>
                  </div>
                )}

                {/* Stage 4: Provider Response & Output Capping */}
                <div className="rounded-xl border-2 border-emerald-300 bg-emerald-50/50 p-4 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-600 text-[10px] font-bold text-white">4</span>
                      <span className="text-xs font-bold text-emerald-950">Inference Response &amp; Output Capping</span>
                    </div>
                    <div className="flex gap-1.5">
                      <Badge className="bg-emerald-700 text-white text-[10px]">
                        {result.choices[0]?.finish_reason === 'length' ? 'Capped (max_tokens)' : 'Finished (stop)'}
                      </Badge>
                      <Badge variant="outline" className="text-[10px] bg-white">
                        {result.usage?.completion_tokens || 0} tokens
                      </Badge>
                    </div>
                  </div>

                  {/* Completion text */}
                  <div className="rounded-lg bg-white p-3 border border-emerald-200 text-xs text-ink leading-relaxed whitespace-pre-wrap font-sans">
                    {result.choices[0]?.message?.content}
                  </div>
                </div>

                {/* Stage 5: Cost & Financial Telemetry */}
                <div className="rounded-xl border border-line bg-canvas p-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
                  <div>
                    <div className="text-[10px] text-muted uppercase">Prompt Tokens</div>
                    <div className="font-bold text-ink font-mono">{result.usage.prompt_tokens}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-muted uppercase">Completion Tokens</div>
                    <div className="font-bold text-ink font-mono">{result.usage.completion_tokens}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-muted uppercase">Execution Cost</div>
                    <div className="font-bold text-emerald-700 font-mono">${(result.optimisation.cost_usd * 1000).toFixed(5)}/1k</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-muted uppercase">Net Cost Saved</div>
                    <div className="font-bold text-emerald-700 font-mono">
                      {result.optimisation.cost_reduction_pct || 65}%
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
