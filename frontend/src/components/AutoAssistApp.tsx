'use client'

import { useEffect, useMemo, useState } from 'react'
import { createClient } from '@supabase/supabase-js'
import ReactMarkdown from 'react-markdown'
import {
  Activity, ArrowRight, ArrowUp, ArrowUpRight, CarFront,
  ChevronRight, CircleAlert, Database, FileSearch, LoaderCircle,
  MessageSquare, Plus, ShieldCheck, Sparkles, UserRound
} from 'lucide-react'

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
  if (typeof window === 'undefined') return 'demo_user'
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
      } catch (e) {
        if (live) setError((e as Error).message)
      }
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
      try { detail = (await response.json()).detail ?? detail } catch { /* leave generic */ }
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

  function newChat() {
    setChatId(null)
    setMessages([])
    setError('')
  }

  async function submit() {
    const message = prompt.trim()
    if (!ready || sending || message.length < 3 || !make.trim() || !model.trim()) return
    setError('')
    setSending(true)
    setPrompt('')
    const optimisticId = crypto.randomUUID()
    setMessages(current => [...current, { id: optimisticId, role: 'user', content: message }])
    try {
      const response = await api('/api/analyze', {
        method: 'POST',
        body: JSON.stringify({
          message,
          make,
          model,
          year,
          ...(chatId ? { conversation_id: chatId } : {}),
        }),
      }) as Analysis
      setChatId(response.conversation_id)
      if (response.generation_error) setError(`AI generation unavailable: ${response.generation_error}`)
      setMessages(current => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer,
          metadata: {
            classification: response.classification,
            sources: response.sources,
            generation_mode: response.generation_mode,
            generation_error: response.generation_error,
            intent: response.intent,
          },
        },
      ])
      const recent = await api('/api/chats') as Chat[]
      setChats(recent)
    } catch (e) {
      setMessages(current => current.filter(item => item.id !== optimisticId))
      setPrompt(message)
      setError((e as Error).message)
    } finally {
      setSending(false)
    }
  }

  const lastAnalysis = useMemo(() => [...messages].reverse().find(m => m.role === 'assistant')?.metadata, [messages])

  return (
    <div className="aa-app app-shell">
      <aside className="aa-nav sidebar">
        <div className="brand aa-brand">
          <div className="brand-mark aa-brand-mark"><CarFront size={22} strokeWidth={2.2} /></div>
          <div>
            <strong>AutoAssist</strong>
            <span className="aa-brand-sub">AUTOMOTIVE INTELLIGENCE</span>
          </div>
        </div>
        <div className="aa-nav-section-label">WORKSPACE</div>
        <button className="new-chat aa-btn-new-chat" onClick={newChat}>
          <Plus size={16} /> New analysis <ChevronRight size={15} />
        </button>
        <div className="aa-history-header">
          <strong>Recent Analyses</strong>
        </div>
        <nav className="history aa-history-list">
          {chats.length === 0 ? (
            <div className="empty-history aa-history-note">No saved analyses yet</div>
          ) : (
            chats.map(chat => (
              <div key={chat.id} className={`aa-history-row ${chat.id === chatId ? 'selected' : ''}`}>
                <button onClick={() => void selectChat(chat.id)} className="aa-history-select">
                  <MessageSquare size={14} />
                  <span>{chat.title}</span>
                </button>
              </div>
            ))
          )}
        </nav>
        <div className="aa-nav-footer">
          <div className="indicator">
            <span className={`aa-online-dot ${health ? 'green' : ''}`} />
            <span>{health ? 'API connected' : 'Connecting to API…'}</span>
          </div>
          <small>Evidence-assisted vehicle investigation. Not a diagnostic service.</small>
        </div>
      </aside>

      <main className="aa-main main-area">
        <header className="aa-topbar topbar">
          <div className="aa-location">
            <span className="aa-location-icon"><Sparkles size={16} /></span>
            <span>Vehicle Analysis</span>
            <ArrowRight size={12} />
            <strong>Direct REST Workspace</strong>
          </div>
          <div className="aa-topbar-right">
            <span className="status-chip aa-runtime-pill">
              <ShieldCheck size={14} /> Evidence-First Mode
            </span>
          </div>
        </header>

        <div className="aa-main-inner page-content">
          <div className="aa-chat-area chat-panel">
            {/* Vehicle context bar */}
            <div className="vehicle-fields aa-vehicle-bar">
              <div className="aa-vehicle-input-group">
                <label>MAKE</label>
                <input value={make} maxLength={60} onChange={e => setMake(e.target.value)} placeholder="Honda" />
              </div>
              <div className="aa-vehicle-input-group">
                <label>MODEL</label>
                <input value={model} maxLength={60} onChange={e => setModel(e.target.value)} placeholder="Accord" />
              </div>
              <div className="aa-vehicle-input-group">
                <label>YEAR</label>
                <input type="number" min={1980} max={2030} value={year} onChange={e => setYear(Number(e.target.value))} />
              </div>
              <div className="aa-vehicle-bar-badge">
                <CarFront size={14} /> Active Target
              </div>
            </div>

            <div className="aa-messages thread" aria-live="polite">
              {messages.length === 0 && !sending && (
                <div className="welcome aa-welcome">
                  <div className="welcome-icon aa-welcome-icon"><Activity size={26} /></div>
                  <div className="aa-kicker">READY TO INVESTIGATE</div>
                  <h1>Understand the issue.<br /><em>Follow the evidence.</em></h1>
                  <p>Classify vehicle complaints, investigate US safety recalls, and review sourced explanations with transparent model outputs.</p>
                  <button className="aa-example-btn" onClick={() => setPrompt(EXAMPLE)}>
                    Try an example complaint <ArrowUpRight size={15} />
                  </button>
                </div>
              )}

              {messages.map(item => {
                const isAssistant = item.role === 'assistant'
                return (
                  <article className={`aa-message message ${item.role}`} key={item.id}>
                    <div className="aa-message-avatar" aria-hidden="true">
                      {isAssistant ? <Sparkles size={18} /> : <UserRound size={16} />}
                    </div>

                    <div className="aa-message-content message-body">
                      <div className="aa-message-name message-role">
                        <span>{isAssistant ? 'AutoAssist' : 'You'}</span>
                        {isAssistant && <span className="aa-verified-tag">Evidence engine</span>}
                      </div>

                      <div className="aa-markdown">
                        <ReactMarkdown>{item.content}</ReactMarkdown>
                      </div>

                      {isAssistant && item.metadata?.sources?.length ? (
                        <div className="inline-sources aa-sources-container">
                          <div className="aa-sources-title">
                            <FileSearch size={14} /> Official NHTSA Sources
                          </div>
                          <div className="aa-source-grid">
                            {item.metadata.sources.map(source => (
                              <a
                                key={source.source_id}
                                href={source.url}
                                target="_blank"
                                rel="noreferrer"
                                className="aa-evidence"
                              >
                                <div className="aa-evidence-top">
                                  <span>NHTSA · {source.source_id}</span>
                                  <ArrowUpRight size={12} />
                                </div>
                                <div className="aa-evidence-title">{source.title}</div>
                                <p>{source.excerpt.slice(0, 150)}…</p>
                              </a>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </article>
                )
              })}

              {sending && (
                <article className="aa-message assistant streaming" aria-label="AutoAssist is analyzing">
                  <div className="aa-message-avatar streaming" aria-hidden="true">
                    <Sparkles size={18} className="aa-spin-pulse" />
                  </div>
                  <div className="aa-message-content">
                    <div className="aa-message-name">
                      <span>AutoAssist</span>
                      <span className="aa-streaming-badge">
                        <span className="aa-stream-live-dot" /> Investigating
                      </span>
                    </div>
                    <div className="processing aa-investigating-pulse">
                      <LoaderCircle size={16} className="aa-spin spin" />
                      <span>Classifying complaint and retrieving NHTSA safety records…</span>
                    </div>
                  </div>
                </article>
              )}
            </div>

            {error && (
              <div className="error-message aa-error" role="alert">
                <CircleAlert size={16} />
                <span>{error}</span>
              </div>
            )}

            <div className="compose aa-composer">
              <textarea
                aria-label="Describe the vehicle complaint"
                value={prompt}
                maxLength={2000}
                placeholder="e.g. My car loses power when accelerating and the check engine light turns on..."
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void submit()
                  }
                }}
              />
              <button
                className="aa-composer-send-btn"
                aria-label="Analyze complaint"
                title="Analyze complaint"
                onClick={() => void submit()}
                disabled={!ready || sending || prompt.trim().length < 3}
              >
                {sending ? <LoaderCircle className="aa-spin spin" size={18} /> : <ArrowUp size={18} />}
              </button>
            </div>
            <div className="hint aa-compose-caption">
              Enter to analyze · Shift+Enter for newline · Sourced from official US vehicle recall data
            </div>
          </div>

          <aside className="aa-side-rail insight-panel">
            <div className="aa-workflow insight-body">
              <div className="aa-panel-heading">
                <div>
                  <div className="aa-card-eyebrow"><Activity size={14} /> MODEL INSPECTION</div>
                  <h3>Model Telemetry</h3>
                  <p className="aa-small">Transparent inference details</p>
                </div>
              </div>

              <div className="aa-inspection-content">
                <div className="metric-label aa-card-eyebrow">PREDICTED COMPONENT</div>
                <div className="classification aa-classification-highlight">
                  {lastAnalysis?.classification?.labels?.length
                    ? lastAnalysis.classification.labels.map(item => item.category).join(', ')
                    : (lastAnalysis?.classification?.category || 'Awaiting input')}
                </div>

                <p className="subtle aa-caption">
                  {lastAnalysis?.classification ?
                    (lastAnalysis.classification.method === 'keyword_demo_only'
                      ? 'Keyword demo only — train a model before claiming ML results.'
                      : `Classifier: ${lastAnalysis.classification.method}`) :
                    'Enter a complaint to trigger the classification model.'}
                </p>

                <div className="aa-v5-evidence-facts metric-grid">
                  <span>
                    <b>{lastAnalysis?.classification?.confidence == null ? '—' : `${(lastAnalysis.classification.confidence * 100).toFixed(1)}%`}</b>
                    <span>Model Confidence</span>
                  </span>
                  <span>
                    <b>{lastAnalysis?.generation_mode?.replaceAll('_', ' ') || '—'}</b>
                    <span>Generation Mode</span>
                  </span>
                </div>

                {lastAnalysis?.intent && (
                  <p className="subtle aa-small">Query route: <strong>{lastAnalysis.intent.replaceAll('_', ' ')}</strong></p>
                )}
                {lastAnalysis?.generation_error && (
                  <p className="subtle aa-agent-error" role="alert">OpenRouter: {lastAnalysis.generation_error}</p>
                )}

                <div className="aa-side-divider" />

                <div className="metric-label aa-card-eyebrow"><Database size={14} /> RETRIEVED NHTSA EVIDENCE</div>
                {lastAnalysis?.sources?.length ? (
                  <div className="aa-evidence-list">
                    {lastAnalysis.sources.map((src, index) => (
                      <a
                        className="evidence-card aa-evidence"
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        key={`${src.source_id}-${index}`}
                      >
                        <div className="aa-evidence-top">
                          <span>NHTSA · {src.source_id}</span>
                          <ArrowUpRight size={13} />
                        </div>
                        <div className="aa-evidence-title">{src.title}</div>
                        <p>{src.excerpt.slice(0, 125)}…</p>
                      </a>
                    ))}
                  </div>
                ) : (
                  <p className="subtle aa-small">No source documents retrieved yet.</p>
                )}
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  )
}
