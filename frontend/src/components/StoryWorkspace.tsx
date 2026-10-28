'use client'

import { FormEvent, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import { useAgent, useCopilotKit } from '@copilotkit/react-core/v2'
import {
  Activity, ArrowRight, ArrowUp, BookOpen, CarFront, CheckCircle2,
  Circle, Clock3, Database, ExternalLink, GitBranch, LoaderCircle,
  MessageCircle, ShieldCheck, Sparkles, StopCircle,
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
type UiState = {
  stage?: string; vehicle?: Vehicle; classification?: Classification;
  recall_status?: string; sources?: Source[]; generation_mode?: string;
  generation_error?: string | null;
}

const FLOW = [
  { id: 'understand_vehicle', label: 'Understand your story', detail: 'Extract vehicle context only when you provide it' },
  { id: 'classify_symptoms', label: 'Classify symptoms', detail: 'Fine-tuned multi-label DistilBERT V3' },
  { id: 'search_nhtsa_recalls', label: 'Check NHTSA', detail: 'Live model-year recall records' },
  { id: 'rank_recall_evidence', label: 'Find relevant evidence', detail: 'Vehicle-scoped hybrid / persistent FAISS' },
  { id: 'compose_grounded_answer', label: 'Generate answer', detail: 'Grounded response with VIN caveats' },
]
const STAGE_PROGRESS: Record<string, number> = {
  queued: 0,
  vehicle_required: 0,
  awaiting_vehicle: 1,
  vehicle_identified: 1,
  classified: 2,
  recalls_checked: 3,
  evidence_retrieved: 4,
  complete: 5,
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
      <div className="aa-evidence-top"><span>NHTSA · {source.source_id}</span><ShieldCheck size={15}/></div>
      <div className="aa-evidence-title">{source.title.replace(/^NHTSA recall \S+ — /, '')}</div>
      <p>{source.excerpt.slice(0, 185)}…</p>
      {url && <a href={url} target="_blank" rel="noreferrer">View original source <ExternalLink size={13}/></a>}
    </article>
  )
}

function ResultsUI({ state }: { state: UiState }) {
  const category = state.classification?.category
  const sources = state.sources || []
  if (!category && !sources.length) return null
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
      <section className="aa-result-card">
        <div className="aa-card-eyebrow"><Database size={14}/> RECALL EVIDENCE</div>
        {sources.length ? <div className="aa-source-grid">{sources.map(s => <EvidenceCard key={s.source_id} source={s}/>)}</div> :
          <p className="aa-small">No sufficiently relevant recall evidence was retrieved. This does not establish that there are no recalls or defects.</p>}
        <div className="aa-caption">Model-year search only. VIN applicability and unrepaired status unverified.</div>
      </section>
    </div>
  )
}

function WorkflowPanel({ state, running }: { state: UiState; running: boolean }) {
  const progress = STAGE_PROGRESS[state.stage || ''] ?? 0
  const awaiting = state.stage === 'awaiting_vehicle'
  return <aside className="aa-workflow">
    <div className="aa-panel-heading"><div><div className="aa-card-eyebrow">LIVE AGENT TRACE</div><h3>Investigation flow</h3></div>
      <span className={running ? 'aa-running' : 'aa-idle'}><span/>{running ? 'Running' : 'Ready'}</span>
    </div>
    <div className="aa-step-list">
      {FLOW.map((step, idx) => {
        const done = progress > idx
        const active = running && progress === idx
        return <div className={`aa-step ${done ? 'done':''} ${active?'active':''}`} key={step.id}>
          <div className="aa-step-indicator">{done ? <CheckCircle2 size={18}/> : active ? <LoaderCircle className="aa-spin" size={18}/> : <Circle size={18}/>}</div>
          <div><strong>{step.label}</strong><p>{step.detail}</p></div>
        </div>
      })}
    </div>
    {awaiting && <div className="aa-info"><MessageCircle size={15}/> Waiting for the vehicle details to continue the investigation.</div>}
    <div className="aa-workflow-footnote">Events reflect completed LangGraph nodes; this first slice does not claim token-level OpenRouter streaming.</div>
  </aside>
}

export default function StoryWorkspace() {
  const { agent } = useAgent({ agentId: AGENT_ID })
  const { copilotkit } = useCopilotKit()
  const [prompt, setPrompt] = useState('')
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)
  const state = (agent.state || {}) as UiState
  const messages = agent.messages.filter(m => m.role === 'user' || m.role === 'assistant')
  const running = sending || agent.isRunning

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages.length, state.stage, agent.isRunning])

  async function send(text: string) {
    const trimmed = text.trim()
    if (!trimmed || running) return
    setError(''); setSending(true); setPrompt('')
    // Reinitialize display progress while preserving the thread's vehicle context.
    agent.setState({ ...agent.state, stage: 'queued', classification: {}, sources: [] })
    agent.addMessage({ id: crypto.randomUUID(), role: 'user', content: trimmed })
    try { await copilotkit.runAgent({ agent }) }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to contact the agent') }
    finally { setSending(false) }
  }

  function onSubmit(event: FormEvent) { event.preventDefault(); void send(prompt) }
  return <div className="aa-app">
    <nav className="aa-nav">
      <Link href="/" className="aa-brand"><span className="aa-brand-mark"><CarFront size={22}/></span><span>autoassist<span className="aa-brand-sub">AUTOMOTIVE INTELLIGENCE</span></span></Link>
      <div className="aa-nav-items"><Link className="aa-nav-active" href="/"><MessageCircle size={16}/> Assistant</Link><Link href="/model-lab"><BookOpen size={16}/> Model Lab</Link></div>
      <div className="aa-nav-footer"><ShieldCheck size={16}/> Evidence-first AI <span>Prototype · Local</span></div>
    </nav>
    <main className="aa-main">
      <header className="aa-topbar"><div className="aa-location"><span className="aa-online-dot"/> AI investigation workspace <ArrowRight size={14}/><strong>Assistant</strong></div><div className="aa-topbar-right"><GitBranch size={15}/> LangGraph + CopilotKit</div></header>
      <div className="aa-main-inner">
        <div className="aa-chat-area">
          {messages.length === 0 && <section className="aa-welcome">
            <div className="aa-welcome-icon"><Sparkles size={26}/></div>
            <div className="aa-kicker">YOUR AUTOMOTIVE AI COPILOT</div>
            <h1>Tell me what<br/><em>happened on the road.</em></h1>
            <p>No forms to fill. Describe what you noticed, and I’ll help investigate the symptoms, identify your vehicle naturally, and check relevant recall evidence.</p>
            <div className="aa-suggestions">{EXAMPLES.map((s, i) => <button key={s} onClick={() => void send(s)}><span>{['01','02','03'][i]}</span>{s}<ArrowUp size={16}/></button>)}</div>
          </section>}
          <div className="aa-messages" aria-live="polite">
            {messages.map((m, idx) => <article key={m.id || idx} className={`aa-message ${m.role}`}>
              <div className="aa-message-avatar">{m.role === 'assistant' ? <Sparkles size={17}/> : 'You'}</div>
              <div className="aa-message-content"><div className="aa-message-name">{m.role === 'assistant' ? 'AutoAssist' : 'You'}</div>
                <div className="aa-markdown"><ReactMarkdown>{messageContent(m.content)}</ReactMarkdown></div>
              </div>
            </article>)}
            {running && <div className="aa-processing"><LoaderCircle size={15} className="aa-spin"/> {state.stage === 'evidence_retrieved' ? 'Preparing evidence-grounded response…' : 'Agent working through your request…'}</div>}
            {!running && state.stage === 'complete' && <ResultsUI state={state}/>}
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
          <WorkflowPanel state={state} running={running}/>
          {state.vehicle && Object.keys(state.vehicle).length > 0 && <section className="aa-context">
            <div className="aa-card-eyebrow"><CarFront size={14}/> VEHICLE CONTEXT</div>
            <strong>{[state.vehicle.year, state.vehicle.make, state.vehicle.model].filter(Boolean).join(' ')}</strong>
            <p>Identified from the conversation, not preselected.</p>
          </section>}
          <section className="aa-method"><div className="aa-card-eyebrow"><Clock3 size={14}/> UNDER THE HOOD</div><strong>See how AutoAssist was built</strong><p>Explore the actual training results, model comparisons, architecture and benchmark limitations.</p><Link href="/model-lab">Explore Model Lab <ArrowRight size={15}/></Link></section>
        </div>
      </div>
    </main>
  </div>
}
