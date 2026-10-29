'use client'

import { FormEvent, useEffect, useRef, useState } from 'react'
import { readLocalThreads, writeLocalThreads, type SavedThread } from './localThreads'
import Link from 'next/link'
import BrandLockup from './BrandLockup'
import ReactMarkdown from 'react-markdown'
import { useAgent, useCopilotKit } from '@copilotkit/react-core/v2'
import {
  Activity, ArrowRight, ArrowUp, BookOpen, CarFront, CheckCircle2,
  Clock3, Database, ExternalLink, LoaderCircle,
  MessageCircle, ShieldCheck, Sparkles, StopCircle, UserRound, Plus, Trash2, FileSearch, Search,
} from 'lucide-react'

const AGENT_ID = 'autoassist'
const EXAMPLES = [
  'The engine loses power when I accelerate, and the check-engine light comes on.',
  'My windshield wipers stop unexpectedly and the reverse camera sometimes goes black.',
  'My brake pedal feels unusually hard and stopping takes longer.',
]

type Vehicle = { make?: string; model?: string; year?: number }
type Label = { category: string; score: number }
type Classification = { category?: string; confidence?: number | null; labels?: Label[]; method?: string }
type Source = { title: string; url: string; source_id: string; excerpt: string }
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
  const url = source.url.startsWith('https://') ? source.url : null
  return (
    <article className="aa-evidence">
      <div className="aa-evidence-top"><span>NHTSA · {source.source_id}</span><FileSearch size={15}/></div>
      <div className="aa-evidence-title">{source.title.replace(/^NHTSA recall \S+ — /, '')}</div>
      <p>{source.excerpt.slice(0, 185)}…</p>
      {url && <a href={url} target="_blank" rel="noreferrer">View original source <ExternalLink size={13}/></a>}
    </article>
  )
}

function ResultsUI({ state }: { state: UiState }) {
  const category = state.classification?.category
  const sources = state.sources || []
  if (!category && state.recall_status === 'not_queried') return null
  return (
    <div className="aa-results">
      {category && <section className="aa-result-card">
        <div className="aa-card-eyebrow"><Activity size={14}/> SYMPTOM CLASSIFICATION</div>
        <div className="aa-classification-row"><strong>{category}</strong>
          {typeof state.classification?.confidence === 'number' &&
            <span>{(state.classification.confidence * 100).toFixed(1)} model score</span>}
        </div>
        {typeof state.classification?.confidence === 'number' &&
          <div className="aa-score-track"><div style={{width:`${Math.min(100, Math.max(0, state.classification.confidence * 100))}%`}}/></div>}
        <div className="aa-caption">Multi-label model output, not a calibrated diagnostic probability.</div>
      </section>}
      {state.recall_status && state.recall_status !== 'not_queried' && <section className="aa-result-card">
        <div className="aa-card-eyebrow"><Database size={14}/> RETRIEVED RECALL CANDIDATES</div>
        {sources.length ? <details className="aa-source-details"><summary>View {sources.length} vehicle-level candidate record{sources.length === 1 ? '' : 's'}</summary><div className="aa-source-grid">{sources.map(s => <EvidenceCard key={s.source_id} source={s}/>)}</div></details> :
          <p className="aa-small">{state.recall_status === 'unavailable' ? 'NHTSA lookup unavailable; please retry or check the official VIN portal.' : 'No sufficiently relevant recall evidence was retrieved; this does not mean no recalls exist.'}</p>}
        <div className="aa-caption">These are research candidates, not confirmed symptom matches. VIN applicability and repair status are unverified.</div>
      </section>}
    </div>
  )
}

function WorkflowPanel({ state, running, failed, steps }: {
  state: UiState; running: boolean; failed: boolean; steps: LiveStep[]
}) {
  const status = running ? 'Working' : failed || state.stage === 'agent_unavailable' ? 'Needs attention' :
    steps.length > 0 || state.stage === 'complete' ? 'Completed' : 'Ready'
  // Only events emitted by this run appear here. No invented five-stage timeline.
  const toolSummaries = new Map((state.tool_events || []).map(item => [item.tool, item.summary]))
  const actualSteps = steps.length ? steps : (state.tool_events || []).map(event => ({
    name: event.tool, status: 'complete' as const, started: 0,
  }))
  return <aside className="aa-workflow">
    <div className="aa-panel-heading"><div><div className="aa-card-eyebrow">LIVE AGENT ACTIVITY</div>
      <h3>Execution trace</h3></div>
      <span className={running ? 'aa-running' : status === 'Completed' ? 'aa-finished' : 'aa-idle'}><span/>{status}</span>
    </div>
    {state.skill && <div className="aa-skill-tag"><Sparkles size={12}/> Skill: {state.skill.replaceAll('_', ' ')}</div>}
    {actualSteps.length ? <div className="aa-step-list">
      {actualSteps.map(step => <div className={`aa-step ${step.status === 'complete' ? 'done' : 'active'}`} key={step.name}>
        <div className="aa-step-indicator">{step.status === 'complete' ? <CheckCircle2 size={18}/> : <LoaderCircle className="aa-spin" size={18}/>}</div>
        <div><strong>{readableStep(step.name)}</strong>
          {toolSummaries.has(step.name) && <p>{toolSummaries.get(step.name)}</p>}
          <span className="aa-step-state">{step.status === 'complete' ? 'Completed' : 'Running'}
            {step.durationMs !== undefined ? ` · ${step.durationMs} ms (client-observed)` : ''}
          </span>
        </div>
      </div>)}
    </div> : <p className="aa-trace-empty">Send a message to see the actual agent nodes run.</p>}
    <div className="aa-workflow-footnote">LangGraph step events and executed tools only. This trace does not reveal hidden model reasoning.</div>
  </aside>
}

export default function StoryWorkspace() {
  const { agent, isReady } = useAgent({ agentId: AGENT_ID })
  const { copilotkit } = useCopilotKit()
  const [prompt, setPrompt] = useState('')
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)
  const [liveSteps, setLiveSteps] = useState<LiveStep[]>([])
  const [streamedText, setStreamedText] = useState('')
  const [threadList, setThreadList] = useState<SavedThread[]>([])
  const [threadQuery, setThreadQuery] = useState('')
  const [activeThread, setActiveThread] = useState('')
  const historyReady = useRef(false)
  const threadRef = useRef<SavedThread[]>([])
  const startedSteps = useRef<Map<string, number>>(new Map())
  const bottom = useRef<HTMLDivElement>(null)
  const state = (agent.state || {}) as UiState
  const messages = agent.messages.filter(m => m.role === 'user' || m.role === 'assistant')
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
      state: { ...agent.state },
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
      onStepStartedEvent: (event: unknown) => {
        const name = eventStepName(event)
        if (!name) return
        startedSteps.current.set(name, Date.now())
        setLiveSteps(previous => previous.some(s => s.name === name) ? previous :
          [...previous, { name, status: 'running', started: Date.now() }])
      },
      onStepFinishedEvent: (event: unknown) => {
        const name = eventStepName(event)
        if (!name) return
        const now = Date.now()
        setLiveSteps(previous => previous.map(s => s.name === name ? {
          ...s, status: 'complete', durationMs: now - (startedSteps.current.get(name) || now),
        } : s))
      },
      onCustomEvent: (payload: unknown) => {
        const p = (payload || {}) as { event?: {name?: string; value?: unknown}; name?: string; value?: unknown }
        const evt = p.event || p
        if (evt.name !== 'autoassist_answer_delta') return
        const data = evt.value as { text?: unknown } | undefined
        const chunk = data?.text
        if (typeof chunk === 'string') setStreamedText(previous => previous + chunk)
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
    setActiveThread(id); setLiveSteps([]); setStreamedText(''); setError('')
    replaceThreads([{ id, title: 'New investigation', updatedAt: Date.now(), messages: [], state: {} }, ...threadRef.current])
  }

  function openThread(entry: SavedThread) {
    if (running || entry.id === activeThread) return
    if (activeThread) saveActive(activeThread)
    agent.threadId = entry.id
    agent.setMessages(entry.messages as Parameters<typeof agent.setMessages>[0])
    agent.setState(entry.state)
    setActiveThread(entry.id); setLiveSteps([]); setStreamedText(''); setError('')
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
    setError(''); setSending(true); setPrompt(''); setStreamedText(''); setLiveSteps([])
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
    finally { saveActive(activeThread); setSending(false); setStreamedText('') }
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
      <header className="aa-topbar"><div className="aa-location"><span className="aa-location-icon"><MessageCircle size={16}/></span><span>Workspace</span><ArrowRight size={13}/><strong>Assistant</strong></div><div className="aa-topbar-right"><span className="aa-runtime-pill"><span className="aa-online-dot"/> LangGraph connected</span></div></header>
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
                    {(snapshot.tool_events || []).length > 0 && <details className="aa-turn-trace"><summary>Executed tools · {snapshot.tool_events?.length}</summary>
                      {(snapshot.tool_events || []).map((event, n) => <div key={n}><strong>{readableStep(event.tool)}</strong><span>{event.summary}</span></div>)}
                    </details>}
                    {snapshot.generation_error && <p className="aa-agent-error">AI service note: {snapshot.generation_error}</p>}
                    {(snapshot.followups || []).length > 0 && <div className="aa-followups">
                      {snapshot.followups?.slice(0, 2).map((suggestion, n) => <button key={n} disabled={running} onClick={() => void send(suggestion)}>{suggestion}<ArrowRight size={13}/></button>)}
                    </div>}
                  </>
                })()}
              </div>
            </article>)}
            {running && <div className="aa-processing">
              {streamedText && messages[messages.length - 1]?.role === 'user' ? <div className="aa-stream-bubble"><div className="aa-card-eyebrow"><span className="aa-stream-live-dot"/><Sparkles size={13}/> AutoAssist · streaming answer</div><div className="aa-markdown"><ReactMarkdown>{streamedText}</ReactMarkdown></div></div>
                : <><LoaderCircle size={15} className="aa-spin"/> {liveSteps.length ? `Agent: ${readableStep(liveSteps[liveSteps.length - 1].name)}…` : 'Preparing your investigation…'}</>}
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
          <WorkflowPanel state={state} running={running} failed={Boolean(error)} steps={liveSteps}/>
          <section className="aa-method"><div className="aa-card-eyebrow"><Clock3 size={14}/> UNDER THE HOOD</div><strong>See how AutoAssist was built</strong><p>Explore the actual training results, model comparisons, architecture and benchmark limitations.</p><Link href="/model-lab">Explore Model Lab <ArrowRight size={15}/></Link></section>
        </div>
      </div>
    </main>
  </div>
}
