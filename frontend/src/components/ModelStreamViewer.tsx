'use client'

import { useEffect, useRef, useState } from 'react'
import { Activity, BrainCircuit, CheckCircle2, ChevronDown, ChevronRight, Cpu, LoaderCircle, Wrench, Terminal, Sparkles } from 'lucide-react'
import type { StreamItem } from './modelStreamEvents'

/** Live in-chat agent stream & tool telemetry viewer, aligned cleanly with the chat stream. */
export default function ModelStreamViewer({ items, running }: { items: StreamItem[]; running: boolean }) {
  const viewport = useRef<HTMLDivElement>(null)
  const [activeTab, setActiveTab] = useState<'all' | 'reasoning' | 'tools'>('all')
  const [isOpen, setIsOpen] = useState(true)

  useEffect(() => {
    if (viewport.current && running) {
      viewport.current.scrollTop = viewport.current.scrollHeight
    }
  }, [items, running])

  const reason = items.filter(e => e.kind === 'reasoning')
  const tools = items.filter(e => e.kind === 'tool-request' || e.kind === 'tool-args' || e.kind === 'tool-result')
  const completedTools = new Set(tools.filter(t => t.kind === 'tool-result').map(t => t.runId))
  const models = items.filter(e => e.kind === 'model')
  const usage = items.filter(e => e.kind === 'usage')
  const chunks = reason.reduce((count, item) => count + item.fragments, 0)
  const toolRequestsCount = tools.filter(t => t.kind === 'tool-request').length

  return (
    <div className="aa-stream-viewer" aria-label="Live agent telemetry and execution stream">
      <div
        className="aa-stream-viewer-header"
        onClick={() => setIsOpen(!isOpen)}
        role="button"
        tabIndex={0}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setIsOpen(!isOpen) } }}
        aria-expanded={isOpen}
      >
        <span className="aa-stream-viewer-icon">
          <BrainCircuit size={16} />
        </span>
        <div className="aa-stream-viewer-title">
          <div className="aa-stream-viewer-title-row">
            <strong>Agent Telemetry &amp; Execution</strong>
            {running ? (
              <span className="aa-live-status-pill running">
                <LoaderCircle size={12} className="aa-spin" /> Live Stream
              </span>
            ) : (
              <span className="aa-live-status-pill finished">
                <CheckCircle2 size={12} /> Execution Complete
              </span>
            )}
          </div>
          <small>{models.at(-1)?.text || 'LangChain agent · streaming telemetry'}</small>
        </div>
        <button
          type="button"
          className="aa-stream-toggle-btn"
          aria-label={isOpen ? 'Collapse telemetry' : 'Expand telemetry'}
        >
          {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
      </div>

      {isOpen && (
        <div className="aa-stream-viewer-body">
          {/* Tab filter bar */}
          <div className="aa-stream-tab-bar">
            <button
              type="button"
              className={`aa-stream-tab-btn ${activeTab === 'all' ? 'active' : ''}`}
              onClick={() => setActiveTab('all')}
            >
              <Terminal size={12} /> Overview
            </button>
            <button
              type="button"
              className={`aa-stream-tab-btn ${activeTab === 'reasoning' ? 'active' : ''}`}
              onClick={() => setActiveTab('reasoning')}
            >
              <Sparkles size={12} /> Reasoning {chunks > 0 && <span className="aa-tab-badge">{chunks}</span>}
            </button>
            <button
              type="button"
              className={`aa-stream-tab-btn ${activeTab === 'tools' ? 'active' : ''}`}
              onClick={() => setActiveTab('tools')}
            >
              <Wrench size={12} /> Tools {toolRequestsCount > 0 && <span className="aa-tab-badge">{toolRequestsCount}</span>}
            </button>
          </div>

          <div className="aa-stream-viewport" ref={viewport} role="log" aria-live="off">
            {/* Reasoning section */}
            {(activeTab === 'all' || activeTab === 'reasoning') && (
              <section className="aa-telemetry-section">
                <div className="aa-telemetry-section-title">
                  <BrainCircuit size={13} />
                  <span>Model Thought Process</span>
                  {chunks > 0 && <small>{chunks} stream chunks</small>}
                </div>
                {reason.length ? (
                  reason.map((item, i) => (
                    <article className="aa-stream-reason-card" key={item.id}>
                      {reason.length > 1 && <div className="aa-stream-pass-label">Pass {i + 1}</div>}
                      <pre>{item.text}</pre>
                      {running && i === reason.length - 1 && <span className="aa-stream-cursor" aria-hidden="true" />}
                    </article>
                  ))
                ) : (
                  <p className="aa-stream-empty-note">
                    {running ? 'Waiting for model reasoning output…' : 'No separate provider reasoning emitted.'}
                  </p>
                )}
              </section>
            )}

            {/* Tools section */}
            {(activeTab === 'all' || activeTab === 'tools') && (
              <section className="aa-telemetry-section">
                <div className="aa-telemetry-section-title">
                  <Wrench size={13} />
                  <span>Tool Execution Pipeline</span>
                  {toolRequestsCount > 0 && <small>{toolRequestsCount} called</small>}
                </div>
                {tools.length ? (
                  <div className="aa-stream-tools-list">
                    {tools.map(item => (
                      <div className={`aa-stream-tool-item kind-${item.kind}`} key={item.id}>
                        <div className="aa-stream-tool-header">
                          <span className="aa-tool-badge-icon">
                            {item.kind === 'tool-request' ? <Wrench size={12} /> : item.kind === 'tool-result' ? <CheckCircle2 size={12} /> : <Activity size={12} />}
                          </span>
                          <strong>{item.kind === 'tool-request' ? `Invoke: ${item.label}` : item.label}</strong>
                          <span className="aa-stream-tool-time">
                            {completedTools.has(item.runId)
                              ? 'Result verified'
                              : running
                              ? 'Executing...'
                              : 'Done'}
                          </span>
                        </div>
                        {item.text && item.kind !== 'tool-request' && (
                          <pre className="aa-stream-tool-payload">{item.text}</pre>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="aa-stream-empty-note">
                    {running ? 'No tools invoked yet.' : 'Direct response · No external tools invoked.'}
                  </p>
                )}
              </section>
            )}

            {/* Usage stats */}
            {usage.length > 0 && (
              <div className="aa-stream-usage-footer">
                <Cpu size={12} />
                <span>{usage.at(-1)?.text}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
