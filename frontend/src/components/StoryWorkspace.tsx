'use client'

import { FormEvent, useEffect, useRef, useState } from 'react'
import { readLocalThreads, writeLocalThreads, type SavedThread } from './localThreads'
import Link from 'next/link'
import BrandLockup from './BrandLockup'
import ModelStreamViewer from './ModelStreamViewer'
import { appendStreamItems, parseAgUiStreamEvent, parseLangChainStream, type StreamItem } from './modelStreamEvents'
import ReactMarkdown from 'react-markdown'
import { useAgent, useCopilotKit } from '@copilotkit/react-core/v2'
import {
  Activity, ArrowRight, ArrowUp, BookOpen, CarFront, CheckCircle2,
  Clock3, Database, ExternalLink, LoaderCircle,
  MessageCircle, ShieldCheck, Sparkles, StopCircle, UserRound, Plus, Trash2, FileSearch, Search, GitBranch,
} from 'lucide-react'

const AGENT_ID = 'autoassist'
const SHOW_PROVIDER_REASONING = process.env.NEXT_PUBLIC_AUTOASSIST_PROVIDER_REASONING === 'true'
const SHOW_STREAM_DEBUG = process.env.NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG === 'true' ||
  (process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG !== 'false')
const EXAMPLES = [
  'The engine loses power when I accelerate, and the check-engine light comes on.',
  'My windshield wipers stop unexpectedly and the reverse camera sometimes goes black.',
  'My brake pedal feels unusually hard and stopping takes longer.',
]

type Vehicle = { make?: string; model?: string; year?: number }
type Label = { category: string; score: number }
type Classification = { category?: string; confidence?: number | null; labels?: Label[]; method?: string }
type Source = { title: string; url: string; source_id: string; excerpt: string }
type EvidenceSummary = { lookup_status?: string; model_year_records?: number; retrieved_candidates?: number; lookup_scope?: string; vin_eligibility_checked?: boolean; repair_status_checked?: boolean }
type ToolEvent = {tool: string; status: string; summary: string}
type UiState = {
  stage?: string
  vehicle?: Vehicle
  skill?: string
  intent?: string
  plan?: {action?: string; skill?: string}
  classification?: Classification
  recall_status?: string
  sources?: Source[]
  generation_mode?: string
  generation_error?: string | null
  tool_events?: ToolEvent[]
  followups?: string[]
  evidence_summary?: EvidenceSummary
  result_history?: (UiState & {turn: number})[]
}

type LiveStep = { name: string; status: 'running' | 'complete'; started: number; durationMs?: number }
function readableStep(name: string): string {
  return name.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase())
}
function eventStepName(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null
  const p = payload as Record<string, unknown>
  const inner = (p.event && typeof p.event === 'object' ? p.event : p) as Record<string, unknown>
  const name = inner.stepName || inner.name
  return typeof name === 'string' && name && !name.startsWith('__') ? name : null
}

function messageContent(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(x =>
    x && typeof x === 'object' && 'text' in x ? String(x.text || '') : ''
  ).join('')
  return ''
}

function EvidenceCard({ source }: { source: Source }) {
  // Only render an official destination; never treat a retrieved URL as trusted HTML.
  let url: string | null = null
  try {
    const destination = new URL(source.url)
    if (destination.protocol === 'https:' && (destination.hostname === 'nhtsa.gov' || destination.hostname.endsWith('.nhtsa.gov'))) url = destination.href
  } catch { /* Invalid source link: show source text without an outgoing link. */ }
  return (
    <article className="aa-evidence">
      <div className="aa-evidence-top"><span>NHTSA · {source.source_id}</span><FileSearch size={15}/></div>
      <div className="aa-evidence-title">{source.title.replace(/^NHTSA recall \S+ — /, '')}</div>
      <p>{source.excerpt.length > 220 ? `${source.excerpt.slice(0, 220)}…` : source.excerpt}</p>
      {url && <a href={url} target="_blank" rel="noreferrer">Read official NHTSA record <ExternalLink size={13}/></a>}
    </article>
  )
}

function ResultsUI({ state }: { state: UiState }) {
  const category = state.classification?.category
  const sources = state.sources || []
  const evidence = state.evidence_summary
  const lookupStatus = evidence?.lookup_status || state.recall_status || 'not_queried'
  const checked = lookupStatus !== 'not_queried'
  if (!category && !checked) return null
  return (
    <div className="aa-results aa-v5-results">
      {category && <section className="aa-result-card">
        <div className="aa-card-eyebrow"><Activity size={14}/> ML MODEL SIGNAL · REPORTED COMPONENT</div>
        <div className="aa-classification-row"><strong>{category}</strong>
          {typeof state.classification?.confidence === 'number' &&
            <span>{(state.classification.confidence * 100).toFixed(1)} model score</span>}
        </div>
        {typeof state.classification?.confidence === 'number' &&
          <div className="aa-score-track" role="meter" aria-label={`${category} model score`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(state.classification.confidence * 100)}><div style={{width:`${Math.min(100, Math.max(0, state.classification.confidence * 100))}%`}}/></div>}
        <div className="aa-caption">Trained on weak owner-complaint labels. This score is neither a calibrated probability nor a mechanical diagnosis.</div>
      </section>}
      {checked && <section className="aa-result-card aa-recall-results">
        <div className="aa-card-eyebrow"><Database size={14}/> NHTSA LOOKUP · MODEL-YEAR SCOPE</div>
        {lookupStatus === 'unavailable' ?
          <p className="aa-v5-evidence-line"><strong>Lookup unavailable.</strong> No conclusion about recalls can be drawn from this attempt.</p> :
          <div className="aa-v5-evidence-facts">
            <span><b>{typeof evidence?.model_year_records === 'number' ? evidence.model_year_records : '—'}</b> model-year records returned</span>
            <span><b>{typeof evidence?.retrieved_candidates === 'number' ? evidence.retrieved_candidates : sources.length}</b> retrieved research candidates</span>
          </div>}
        {sources.length ? <details className="aa-source-details"><summary>Inspect {sources.length} candidate source{sources.length === 1 ? '' : 's'}</summary><div className="aa-source-grid">{sources.map((source, index) => <EvidenceCard key={`${source.source_id}-${index}`} source={source}/>)}</div></details> :
          lookupStatus !== 'unavailable' && <p className="aa-small">No sufficiently relevant candidate was selected by the retriever. This does not establish that the vehicle has no recalls.</p>}
        <div className="aa-v5-evidence-limit"><ShieldCheck size={14}/><span>Retrieved documents are <strong>not verified causes</strong>. Make/model/year matching does not establish VIN eligibility or whether a repair is outstanding.</span></div>
      </section>}
    </div>
  )
}

function TurnTrace({ steps, tools }: { steps: LiveStep[]; tools: ToolEvent[] }) {
  if (!steps.length && !tools.length) return null
  const toolSummaries = new Map(tools.map(event => [event.tool, event]))
  return <details className="aa-turn-trace aa-v5-turn-trace">
    <summary><GitBranch size={14}/> View agent steps <span>{steps.length ? `${steps.length} observed` : `${tools.length} tools`}</span></summary>
    <div className="aa-v5-trace-content">
      {steps.length > 0 ? steps.map((step, idx) => {
        const tool = toolSummaries.get(step.name)
        return <div className="aa-v5-trace-row" key={`${step.name}-${idx}`}>
          {step.status === 'complete' ? <CheckCircle2 size={15}/> : <LoaderCircle size={15}/>}
          <div><strong>{readableStep(step.name)}</strong>{tool && <p>{tool.summary}</p>}</div>
          {step.durationMs !== undefined && <small>{step.durationMs} ms</small>}
        </div>
      }) : tools.map((tool, idx) => <div className="aa-v5-trace-row" key={`${tool.tool}-${idx}`}><CheckCircle2 size={15}/><div><strong>{readableStep(tool.tool)}</strong><p>{tool.summary}</p></div></div>)}
    </div>
  </details>
}

function GuidePanel() {
  return <section className="aa-v6-guide" aria-label="Ways to investigate">
    <div className="aa-card-eyebrow"><Sparkles size={14}/> YOUR INVESTIGATION</div>
    <h3>Ask naturally. Go deeper.</h3>
    <p>Describe a symptom, ask about a warning light, or check published recalls after sharing your vehicle.</p>
    <div className="aa-v6-guide-row"><Activity size={15}/> Understand reported symptoms</div>
    <div className="aa-v6-guide-row"><FileSearch size={15}/> Inspect official recall sources</div>
    <div className="aa-v6-guide-row"><ShieldCheck size={15}/> Separate evidence from possibilities</div>
    <small>AutoAssist supports investigation; it cannot diagnose a vehicle or verify outstanding VIN repairs.</small>
  </section>
}

// Render only observed nodes; never fabricate tool calls or duplicate this timeline in the sidebar.
function LiveActivity({ steps }: { steps: LiveStep[] }) {
  const completed = steps.filter(step => step.status === 'complete').length
  const active = [...steps].reverse().find(step => step.status === 'running')
  return <details className="aa-v6-live-activity" aria-label="Observed agent execution">
    <summary>
      <span className="aa-v6-live-icon"><LoaderCircle size={15} className="aa-spin"/></span>
      <span className="aa-v6-live-title">{active ? readableStep(active.name) : 'Preparing the investigation'}</span>
      <span className="aa-v6-live-meta">{steps.length ? `${completed}/${steps.length} observed steps` : 'Connecting to agent'}</span>
      <span className="aa-v6-live-chevron">Details</span>
    </summary>
    <div className="aa-v6-live-detail">
      {steps.length === 0 ? <p>Waiting for the agent to emit its first execution event.</p> :
        steps.map(step => <div key={step.name} className="aa-v6-live-step">
          {step.status === 'complete' ? <CheckCircle2 size={14}/> : <LoaderCircle size={14} className="aa-spin"/>}
          <span>{readableStep(step.name)}</span>
          {step.durationMs !== undefined && <small>{step.durationMs} ms</small>}
        </div>)}
    </div>
  </details>
}

export default function StoryWorkspace() {
  const { agent, isReady } = useAgent({ agentId: AGENT_ID })
  const { copilotkit } = useCopilotKit()
  const [prompt, setPrompt] = useState('')
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)
  const [liveSteps, setLiveSteps] = useState<LiveStep[]>([])
  const liveStepsRef = useRef<LiveStep[]>([])
  const [stepsByTurn, setStepsByTurn] = useState<Record<string, LiveStep[]>>({})
  const stepsByTurnRef = useRef<Record<string, LiveStep[]>>({})
  const [streamedText, setStreamedText] = useState('')
  const [providerReasoning, setProviderReasoning] = useState('')
  const [streamEvents, setStreamEvents] = useState<StreamItem[]>([])
  const [threadList, setThreadList] = useState<SavedThread[]>([])
  const [threadQuery, setThreadQuery] = useState('')
  const [activeThread, setActiveThread] = useState('')
  const historyReady = useRef(false)
  const threadRef = useRef<SavedThread[]>([])
  const startedSteps = useRef<Map<string, number>>(new Map())
  const bottom = useRef<HTMLDivElement>(null)
  const state = (agent.state || {}) as UiState
  const messages = agent.messages.filter(m => m.role === 'user' || (m.role === 'assistant' && messageContent(m.content).trim().length > 0))
  const running = sending || agent.isRunning

  function replaceThreads(next: SavedThread[]) {
    const sorted = [...next].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 30)
    threadRef.current = sorted
    setThreadList(sorted)
    writeLocalThreads(sorted)
  }

  function saveActive(id: string) {
    const t = threadRef.current.find(x => x.id === id)
    if (!t) return
    replaceThreads(threadRef.current.map(x => x.id === id ? {
      ...x, updatedAt: Date.now(), messages: agent.messages.map(m => ({...m})),
      state: { ...agent.state }, stepsByTurn: { ...stepsByTurnRef.current },
    } : x))
  }

  // This local-only thread list is explicit: the current InMemoryAgentRunner has
  // no durable CopilotKit history service. Never market this as cloud persistence.
  useEffect(() => {
    if (!isReady || historyReady.current) return
    historyReady.current = true
    const saved = readLocalThreads()
    if (saved.length) {
      threadRef.current = saved
      setThreadList(saved)
      const selected = saved[0]
      agent.threadId = selected.id
      agent.setMessages(selected.messages as Parameters<typeof agent.setMessages>[0])
      agent.setState(selected.state)
      setActiveThread(selected.id)
      stepsByTurnRef.current = selected.stepsByTurn || {}
      setStepsByTurn(stepsByTurnRef.current)
    } else {
      const id = agent.threadId || crypto.randomUUID()
      agent.threadId = id
      setActiveThread(id)
      replaceThreads([{ id, title: 'New investigation', updatedAt: Date.now(), messages: [], state: {} }])
    }
  }, [agent, isReady])

  useEffect(() => {
    if (!isReady) return
    const subscription = agent.subscribe({
      onRawEvent: (payload: unknown) => {
        if (SHOW_STREAM_DEBUG) {
          const entries = parseLangChainStream(payload)
          if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
        }
        if (!SHOW_PROVIDER_REASONING) return
        // create_agent emits native on_chat_model_stream events. In a RAW
        // LangChain event only provider-exposed reasoning blocks are displayed.
        // Never synthesize reasoning or concatenate it with answer text.
        const envelope = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {}
        const raw = (envelope.rawEvent && typeof envelope.rawEvent === 'object' ? envelope.rawEvent :
          envelope.event && typeof envelope.event === 'object' ? envelope.event : envelope) as Record<string, unknown>
        if (raw.event !== 'on_chat_model_stream') return
        const data = raw.data && typeof raw.data === 'object' ? raw.data as Record<string, unknown> : {}
        const chunk = data.chunk && typeof data.chunk === 'object' ? data.chunk as Record<string, unknown> : {}
        const blocks = (Array.isArray(chunk.content_blocks) ? chunk.content_blocks :
          Array.isArray(chunk.content) ? chunk.content : []) as unknown[]
        let foundBlock = false
        for (const block of blocks) {
          if (!block || typeof block !== 'object') continue
          const b = block as Record<string, unknown>
          if (b.type !== 'reasoning') continue
          const value = typeof b.reasoning === 'string' ? b.reasoning :
            typeof b.text === 'string' ? b.text : ''
          if (value && !value.includes('[REDACTED]')) {
            foundBlock = true
            setProviderReasoning(previous => (previous + value).slice(0, 6000))
          }
        }
        // Older OpenRouter/LangChain versions sometimes place reasoning in
        // provider-specific metadata instead of normalized content_blocks.
        if (!foundBlock) {
          const extra = chunk.additional_kwargs && typeof chunk.additional_kwargs === 'object' ?
            chunk.additional_kwargs as Record<string, unknown> : {}
          let providerText = typeof extra.reasoning === 'string' ? extra.reasoning : ''
          if (!providerText && Array.isArray(extra.reasoning_details)) {
            providerText = extra.reasoning_details.map(item => {
              if (!item || typeof item !== 'object') return ''
              const part = item as Record<string, unknown>
              if (part.type !== 'reasoning.text' && part.type !== 'reasoning.summary') return ''
              return typeof part.text === 'string' ? part.text :
                typeof part.summary === 'string' ? part.summary : ''
            }).join('')
          }
          if (providerText && !providerText.includes('[REDACTED]')) {
            setProviderReasoning(previous => (previous + providerText).slice(0, 6000))
          }
        }
      },
      // The backend emits *native* AG-UI reasoning + tool events alongside RAW.
      // Subscribe here rather than relying on RAW decoding or an env-only panel.
      onReasoningMessageContentEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onToolCallStartEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onToolCallArgsEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onToolCallResultEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onStepStartedEvent: (event: unknown) => {
        const name = eventStepName(event)
        if (!name) return
        const now = Date.now()
        startedSteps.current.set(name, now)
        const next = liveStepsRef.current.some(step => step.name === name) ? liveStepsRef.current :
          [...liveStepsRef.current, { name, status: 'running' as const, started: now }]
        liveStepsRef.current = next
        setLiveSteps(next)
      },
      onStepFinishedEvent: (event: unknown) => {
        const name = eventStepName(event)
        if (!name) return
        const now = Date.now()
        const next = liveStepsRef.current.map(step => step.name === name ? {
          ...step, status: 'complete' as const, durationMs: now - (startedSteps.current.get(name) || now),
        } : step)
        liveStepsRef.current = next
        setLiveSteps(next)
      },
      onCustomEvent: (payload: unknown) => {
        const p = (payload || {}) as { event?: {name?: string; value?: unknown}; name?: string; value?: unknown }
        const evt = p.event || p
        if (evt.name !== 'autoassist_answer_delta' && evt.name !== 'autoassist_provider_reasoning_delta') return
        const data = evt.value as { text?: unknown } | undefined
        const chunk = data?.text
        if (typeof chunk !== 'string') return
        if (evt.name === 'autoassist_provider_reasoning_delta') {
          if (!SHOW_PROVIDER_REASONING) return
          setProviderReasoning(previous => (previous + chunk).slice(0, 6000))
        } else {
          setStreamedText(previous => previous + chunk)
        }
      },
    })
    return () => subscription.unsubscribe()
  }, [agent, isReady])

  function newThread() {
    if (running) return
    if (activeThread) saveActive(activeThread)
    const id = crypto.randomUUID()
    agent.threadId = id
    agent.setMessages([])
    agent.setState({})
    liveStepsRef.current = []; stepsByTurnRef.current = {}
    setStepsByTurn({}); setActiveThread(id); setLiveSteps([]); setStreamedText(''); setProviderReasoning(''); setStreamEvents([]); setError('')
    replaceThreads([{ id, title: 'New investigation', updatedAt: Date.now(), messages: [], state: {} }, ...threadRef.current])
  }

  function openThread(entry: SavedThread) {
    if (running || entry.id === activeThread) return
    if (activeThread) saveActive(activeThread)
    agent.threadId = entry.id
    agent.setMessages(entry.messages as Parameters<typeof agent.setMessages>[0])
    agent.setState(entry.state)
    liveStepsRef.current = []; stepsByTurnRef.current = entry.stepsByTurn || {}
    setStepsByTurn(stepsByTurnRef.current); setActiveThread(entry.id); setLiveSteps([]); setStreamedText(''); setProviderReasoning(''); setStreamEvents([]); setError('')
  }

  function deleteThread(id: string) {
    if (running) return
    replaceThreads(threadRef.current.filter(t => t.id !== id))
    if (activeThread === id) {
      const other = threadRef.current.find(t => t.id !== id)
      if (other) openThread(other)
      else {
        setActiveThread('')
        newThread()
      }
    }
  }

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages.length, state.stage, agent.isRunning])

  async function send(text: string) {
    const trimmed = text.trim()
    if (!trimmed || running) return
    setError(''); setSending(true); setPrompt(''); setStreamedText(''); setProviderReasoning(''); setStreamEvents([]); setLiveSteps([])
    liveStepsRef.current = []
    const userTurn = messages.filter(message => message.role === 'user').length + 1
    startedSteps.current.clear()
    const thread = threadRef.current.find(t => t.id === activeThread)
    if (thread && thread.title === 'New investigation') {
      replaceThreads(threadRef.current.map(t => t.id === activeThread ?
        { ...t, title: trimmed.slice(0, 48) + (trimmed.length > 48 ? '…' : '') } : t))
    }
    // Reinitialize display progress while preserving the thread's vehicle context.
    agent.setState({ ...agent.state, stage: 'queued', plan: {}, skill: '', intent: '', classification: {}, sources: [], tool_events: [], followups: [], recall_status: 'not_queried', generation_error: null })
    agent.addMessage({ id: crypto.randomUUID(), role: 'user', content: trimmed })
    try { await copilotkit.runAgent({ agent }) }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to contact the agent') }
    finally {
      const next = { ...stepsByTurnRef.current, [String(userTurn)]: liveStepsRef.current }
      stepsByTurnRef.current = next
      setStepsByTurn(next)
      saveActive(activeThread)
      setSending(false); setStreamedText('')
    }
  }

  function onSubmit(event: FormEvent) { event.preventDefault(); void send(prompt) }
  return <div className="aa-app">
    <nav className="aa-nav">
      <BrandLockup/>
      <div className="aa-nav-section-label">WORKSPACE</div>
      <div className="aa-nav-items"><Link className="aa-nav-active" href="/" aria-current="page"><MessageCircle size={17}/> Assistant</Link><Link href="/model-lab"><BookOpen size={17}/> Model Lab</Link></div>
      <div className="aa-history-header"><strong>Investigations</strong><button onClick={newThread} disabled={running} title="New investigation" aria-label="New investigation"><Plus size={17}/></button></div>
      <label className="aa-history-search"><Search size={15}/><input aria-label="Search conversations" value={threadQuery} onChange={event => setThreadQuery(event.target.value)} placeholder="Search conversations"/></label>
      <div className="aa-history-list">{threadList.filter(thread => thread.title.toLowerCase().includes(threadQuery.toLowerCase())).map(thread => <div className={`aa-history-row ${thread.id === activeThread ? 'selected' : ''}`} key={thread.id}>
        <button className="aa-history-select" onClick={() => openThread(thread)} disabled={running} title={thread.title}>{thread.title}</button>
        <button className="aa-history-delete" onClick={() => deleteThread(thread.id)} disabled={running} aria-label={`Delete ${thread.title}`} title="Delete saved conversation"><Trash2 size={13}/></button>
      </div>)}</div>
      <div className="aa-history-note">Saved in this browser for local development only.</div>
      <div className="aa-nav-footer"><span className="aa-footer-icon"><ShieldCheck size={17}/></span><div><strong>Evidence-first AI</strong><small>Prototype · Local workspace</small></div></div>
    </nav>
    <main className="aa-main">
      <header className="aa-topbar"><div className="aa-location"><span className="aa-location-icon"><MessageCircle size={16}/></span><span>Workspace</span><ArrowRight size={13}/><strong>Assistant</strong></div><div className="aa-topbar-right"><span className="aa-runtime-pill"><span className="aa-online-dot"/> LangChain agent</span></div></header>
      <div className="aa-main-inner">
        <div className="aa-chat-area">
          {state.vehicle && Object.keys(state.vehicle).length > 0 && <div className="aa-vehicle-chip"><CarFront size={14}/> {[state.vehicle.year, state.vehicle.make, state.vehicle.model].filter(Boolean).join(' ')}</div>}
          {messages.length === 0 && <section className="aa-welcome">
            <div className="aa-welcome-icon"><Sparkles size={26}/></div>
            <div className="aa-kicker">YOUR AUTOMOTIVE AI COPILOT</div>
            <h1>Tell me what<br/><em>happened on the road.</em></h1>
            <p>No forms to fill. Describe what you noticed, and I’ll help investigate the symptoms, identify your vehicle naturally, and check relevant recall evidence.</p>
            <div className="aa-suggestions">{EXAMPLES.map((s, i) => <button key={s} onClick={() => void send(s)}><span>{['01','02','03'][i]}</span>{s}<ArrowUp size={16}/></button>)}</div>
          </section>}
          <div className="aa-messages" aria-live="polite">
            {messages.map((m, idx) => <article key={m.id || idx} className={`aa-message ${m.role}`}>
              <div className="aa-message-avatar" aria-hidden="true">{m.role === 'assistant' ? <Sparkles size={17}/> : <UserRound size={16}/>}</div>
              <div className="aa-message-content"><div className="aa-message-name">{m.role === 'assistant' ? 'AutoAssist' : 'You'}</div>
                <div className="aa-markdown"><ReactMarkdown>{messageContent(m.content)}</ReactMarkdown></div>
                {m.role === 'assistant' && (() => {
                  const userTurn = messages.slice(0, idx).filter(x => x.role === 'user').length
                  const snapshot = state.result_history?.find(x => x.turn === userTurn)
                  if (!snapshot) return null
                  return <>
                    <ResultsUI state={snapshot}/>
                    <TurnTrace steps={stepsByTurn[String(userTurn)] || []} tools={snapshot.tool_events || []}/>
                    {snapshot.generation_error && <p className="aa-agent-error">AI service note: {snapshot.generation_error}</p>}
                    {(snapshot.followups || []).length > 0 && <div className="aa-followups">
                      {snapshot.followups?.slice(0, 2).map((suggestion, n) => <button key={n} disabled={running} onClick={() => void send(suggestion)}>{suggestion}<ArrowRight size={13}/></button>)}
                    </div>}
                  </>
                })()}
              </div>
            </article>)}
            {running && !SHOW_STREAM_DEBUG && <LiveActivity steps={liveSteps}/>}
            {SHOW_STREAM_DEBUG && (running || streamEvents.length > 0) && <ModelStreamViewer items={streamEvents} running={running}/>}
            {!SHOW_STREAM_DEBUG && providerReasoning && <details className="aa-provider-reasoning" aria-label="OpenRouter provider reasoning">
              <summary><Sparkles size={14}/> Provider reasoning <span>OpenRouter · available text</span></summary>
              <div className="aa-provider-reasoning-text">{providerReasoning}</div>
            </details>}
            {running && streamedText && messages[messages.length - 1]?.role === 'user' &&
              <div className="aa-processing aa-v6-streaming" aria-label="Streaming assistant response">
                <div className="aa-stream-bubble">
                  <div className="aa-card-eyebrow"><span className="aa-stream-live-dot"/><Sparkles size={13}/> AutoAssist · streaming answer</div>
                  <div className="aa-markdown"><ReactMarkdown>{streamedText}</ReactMarkdown></div>
                  <p className="aa-v5-draft-caption">Streaming text is provisional until the agent finishes evidence checks.</p>
                </div>
              </div>}
            <div ref={bottom}/>
          </div>
          {error && <div className="aa-error" role="alert">{error}</div>}
          <form className="aa-composer" onSubmit={onSubmit}>
            <textarea value={prompt} disabled={running} onChange={e => setPrompt(e.target.value)} placeholder="Describe what happened with your car…" rows={2} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(prompt) } }}/>
            {running ? <button type="button" aria-label="Stop generation" onClick={() => copilotkit.stopAgent({agent})}><StopCircle size={19}/></button>
              : <button type="submit" aria-label="Send message" disabled={!prompt.trim()}><ArrowUp size={19}/></button>}
          </form>
          <div className="aa-compose-caption"><ShieldCheck size={12}/> Decision support only · Vehicle recalls are not VIN-specific · Shift+Enter for a new line</div>
        </div>
        <div className="aa-side-rail">
          <GuidePanel/>
          <section className="aa-method"><div className="aa-card-eyebrow"><Clock3 size={14}/> UNDER THE HOOD</div><strong>See how AutoAssist was built</strong><p>Explore the actual training results, model comparisons, architecture and benchmark limitations.</p><Link href="/model-lab">Explore Model Lab <ArrowRight size={15}/></Link></section>
        </div>
      </div>
    </main>
  </div>
}
