'use client'

import { useEffect, useMemo, useState } from 'react'
import { createClient } from '@supabase/supabase-js'
import ReactMarkdown from 'react-markdown'
import { Activity, ArrowUpRight, CarFront, ChevronRight, CircleAlert, Database, LoaderCircle, MessageSquare, Plus, Send, ShieldCheck, Sparkles } from 'lucide-react'

type Classification = { category: string; confidence: number | null; method: string; labels?: { category: string; score: number }[] }
type Source = { title: string; url: string; source_id: string; excerpt: string }
type Analysis = {
  conversation_id: string; answer: string; classification: Classification
  sources: Source[]; recall_status: string; generation_mode: string
  generation_error?: string | null; intent?: string
}
type Chat = { id: string; title: string; created_at: string }
type ChatMessage = { id: string; role: 'user' | 'assistant'; content: string; metadata?: { classification?: Classification; sources?: Source[]; generation_mode?: string; generation_error?: string | null; intent?: string } }
type Health = { status: string; model: string; storage: string; llm_configured: boolean }

const API_URL = (process.env.NEXT_PUBLIC_API_URL)?.replace(/\/$/, '') || 'http://localhost:8000'
const supaUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supaKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
const supabase = supaUrl && supaKey ? createClient(supaUrl, supaKey) : null
const EXAMPLE = 'The vehicle loses power while accelerating and the check-engine light comes on. What should I investigate?'

function createLocalId(): string {
  let id = localStorage.getItem('autoassist_demo_user')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('autoassist_demo_user', id)
  }
  return id
}

export default function AutoAssistApp() {
  const [token, setToken] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const [chats, setChats] = useState<Chat[]>([])
  const [chatId, setChatId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [make, setMake] = useState('Honda')
  const [model, setModel] = useState('Accord')
  const [year, setYear] = useState(2018)
  const [prompt, setPrompt] = useState('')
  const [sending, setSending] = useState(false)

  useEffect(() => {
    let live = true
    async function boot() {
      try {
        if (supabase) {
          const existing = await supabase.auth.getSession()
          if (existing.error) throw existing.error
          const session = existing.data.session ?? (await supabase.auth.signInAnonymously()).data.session
          if (!session) throw new Error('Could not create an anonymous Supabase session. Enable Anonymous Sign-Ins in Supabase Auth.')
          if (live) setToken(session.access_token)
        } else {
          if (live) setToken(createLocalId())
        }
        if (live) setReady(true)
      } catch (e) { if (live) setError((e as Error).message) }
    }
    void boot()
    fetch(`${API_URL}/api/health`).then(r => r.json()).then(setHealth).catch(() => {})
    return () => { live = false }
  }, [])

  async function authHeaders() {
    if (supabase) {
      const { data, error: sessionError } = await supabase.auth.getSession()
      if (sessionError || !data.session) throw new Error('Supabase login expired; reload the page.')
      return { Authorization: `Bearer ${data.session.access_token}` }
    }
    return { 'X-Demo-User': token || createLocalId() }
  }

  async function api(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers)
    headers.set('Content-Type', 'application/json')
    for (const [name, value] of Object.entries(await authHeaders())) {
      if (value !== undefined) headers.set(name, value)
    }
    const response = await fetch(`${API_URL}${path}`, { ...init, headers })
    if (!response.ok) {
      let detail = `Request failed (${response.status})`
      try { detail = (await response.json()).detail ?? detail } catch { /* leave generic error */ }
      throw new Error(detail)
    }
    return response.json()
  }

  useEffect(() => {
    if (!ready) return
    void api('/api/chats').then((items: Chat[]) => setChats(items)).catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready])

  async function selectChat(id: string) {
    setError('')
    try {
      const records = await api(`/api/chats/${id}/messages`) as ChatMessage[]
      setChatId(id)
      setMessages(records)
    } catch (e) { setError((e as Error).message) }
  }

  function newChat() { setChatId(null); setMessages([]); setError('') }

  async function submit() {
    const message = prompt.trim()
    if (!ready || sending || message.length < 3 || !make.trim() || !model.trim()) return
    setError(''); setSending(true); setPrompt('')
    const optimisticId = crypto.randomUUID()
    setMessages(current => [...current, { id: optimisticId, role: 'user', content: message }])
    try {
      const response = await api('/api/analyze', {
        method: 'POST', body: JSON.stringify({ message, make, model, year,
          ...(chatId ? { conversation_id: chatId } : {}) }),
      }) as Analysis
      setChatId(response.conversation_id)
      if (response.generation_error) setError(`AI generation unavailable: ${response.generation_error}`)
      setMessages(current => [...current, {
        id: crypto.randomUUID(), role: 'assistant', content: response.answer,
        metadata: { classification: response.classification,
          sources: response.sources, generation_mode: response.generation_mode,
          generation_error: response.generation_error, intent: response.intent },
      }])
      const recent = await api('/api/chats') as Chat[]
      setChats(recent)
    } catch (e) {
      setMessages(current => current.filter(item => item.id !== optimisticId))
      setPrompt(message)
      setError((e as Error).message)
    } finally { setSending(false) }
  }

  const lastAnalysis = useMemo(() => [...messages].reverse().find(m => m.role === 'assistant')?.metadata, [messages])
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><CarFront size={22} strokeWidth={2.2} /></div>
          <div><strong>AutoAssist</strong><span>AUTOMOTIVE INTELLIGENCE</span></div></div>
        <div className="side-label">WORKSPACE</div>
        <button className="new-chat" onClick={newChat}><Plus size={17}/> New analysis <ChevronRight size={16}/></button>
        <div className="side-label history-label">RECENT ANALYSES</div>
        <nav className="history">
          {chats.length === 0 ? <div className="empty-history">No saved analyses yet</div> : chats.map(chat => (
            <button key={chat.id} onClick={() => void selectChat(chat.id)} className={`history-item ${chat.id === chatId ? 'selected' : ''}`}>
              <MessageSquare size={16}/><span>{chat.title}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="indicator"><span className={`dot ${health ? 'green' : ''}`}/>{health ? 'API connected' : 'API unavailable'}</div>
          <p>Prototype for evidence-assisted vehicle investigation. Not a diagnostic service.</p>
        </div>
      </aside>
      <main className="main-area">
        <header className="topbar"><div><span className="eyebrow">RESEARCH PROTOTYPE</span><h1>Vehicle analysis <span className="muted-title">/ workspace</span></h1></div>
          <div className="status-chip"><ShieldCheck size={14}/> Evidence-first AI</div></header>
        <div className="page-content">
          <section className="hero"><div className="hero-icon"><Sparkles size={23}/></div>
            <div><div className="eyebrow amber">MEET YOUR ASSISTANT</div><h2>Understand the issue.<br/><em>Follow the evidence.</em></h2>
              <p>Classify vehicle complaints, investigate US safety recalls, and review sourced explanations with transparent model outputs.</p></div>
          </section>
          <section className="workspace-grid">
            <div className="chat-panel">
              <div className="section-heading"><div><h3>Complaint analysis</h3><p>Describe the issue in your own words.</p></div><span className="live-tag"><span className="dot green"/> LIVE WORKFLOW</span></div>
              <div className="vehicle-fields">
                <label>MAKE<input value={make} maxLength={60} onChange={e => setMake(e.target.value)} placeholder="Honda" /></label>
                <label>MODEL<input value={model} maxLength={60} onChange={e => setModel(e.target.value)} placeholder="Accord" /></label>
                <label>YEAR<input type="number" min={1980} max={2030} value={year} onChange={e => setYear(Number(e.target.value))} /></label>
              </div>
              <div className="thread" aria-live="polite">
                {messages.length === 0 && <div className="welcome">
                  <Activity size={26}/><strong>Ready to investigate</strong>
                  <span>Enter a vehicle complaint to see classification, relevant recalls, and a sourced response.</span>
                  <button onClick={() => setPrompt(EXAMPLE)}>Try an example complaint <ArrowUpRight size={15}/></button>
                </div>}
                {messages.map(item => <div className={`message ${item.role}`} key={item.id}>
                  <div className="message-role">{item.role === 'assistant' ? 'AUTOASSIST' : 'YOU'}</div>
                  <div className="message-body"><ReactMarkdown>{item.content}</ReactMarkdown></div>
                  {item.role === 'assistant' && item.metadata?.sources?.length ? <div className="inline-sources">
                    {item.metadata.sources.map(source => <a key={source.source_id} href={source.url} target="_blank" rel="noreferrer">NHTSA {source.source_id} <ArrowUpRight size={12}/></a>)}
                  </div> : null}
                </div>)}
                {sending && <div className="processing"><LoaderCircle size={17} className="spin"/> Classifying and checking NHTSA records...</div>}
              </div>
              {error && <div className="error-message" role="alert"><CircleAlert size={16}/>{error}</div>}
              <div className="compose"><textarea aria-label="Describe the vehicle complaint" value={prompt} maxLength={2000}
                placeholder="e.g. My car loses power when accelerating and the check engine light turns on..."
                onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void submit() } }} />
                <button aria-label="Analyze complaint" title="Analyze complaint" onClick={() => void submit()} disabled={!ready || sending || prompt.trim().length < 3}>
                  {sending ? <LoaderCircle className="spin" size={19}/> : <Send size={19}/>}</button></div>
              <div className="hint">Enter to analyze · Shift+Enter for newline · US vehicle recall data</div>
            </div>
            <aside className="insight-panel">
              <div className="section-heading"><div><h3>Model inspection</h3><p>Transparent inference details</p></div><Activity size={19}/></div>
              <div className="insight-body">
                <div className="metric-label">CLASSIFICATION</div>
                <div className="classification">{lastAnalysis?.classification?.labels?.length
                  ? lastAnalysis.classification.labels.map(item => item.category).join(', ')
                  : (lastAnalysis?.classification?.category || 'Awaiting input')}</div>
                <p className="subtle">{lastAnalysis?.classification ?
                  (lastAnalysis.classification.method === 'keyword_demo_only' ? 'Keyword demo only — train a model before claiming ML results.' : `Model: ${lastAnalysis.classification.method}`) :
                  'A trained model will replace the fallback classifier.'}</p>
                <p className="subtle">Classifier scores are not calibrated probabilities or confirmed mechanical diagnoses.</p>
                <div className="metric-grid"><div><span>MODEL SCORE</span><strong>{lastAnalysis?.classification?.confidence == null ? '—' : `${(lastAnalysis.classification.confidence * 100).toFixed(1)}%`}</strong></div>
                  <div><span>LLM MODE</span><strong className="small-metric">{lastAnalysis?.generation_mode?.replaceAll('_', ' ') || '—'}</strong></div></div>
                {lastAnalysis?.intent && <p className="subtle">Query route: {lastAnalysis.intent.replaceAll('_', ' ')}</p>}
                {lastAnalysis?.generation_error && <p className="subtle" role="alert">OpenRouter: {lastAnalysis.generation_error}</p>}
                <div className="side-divider"/>
                <div className="metric-label">RETRIEVED EVIDENCE</div>
                {lastAnalysis?.sources?.length ? lastAnalysis.sources.map((src, index) => <a className="evidence-card" href={src.url} target="_blank" rel="noreferrer" key={`${src.source_id}-${index}`}>
                  <div><Database size={15}/> NHTSA <ArrowUpRight size={13}/></div><strong>{src.title}</strong><p>{src.excerpt.slice(0, 125)}...</p></a>) : <p className="subtle">No source documents retrieved yet.</p>}
              </div>
            </aside>
          </section>
          <footer className="footer"><span>AutoAssist · Applied ML & GenAI portfolio project</span><span>Owner reports and recall matches are not confirmed diagnoses.</span></footer>
        </div>
      </main>
    </div>
  )
}
