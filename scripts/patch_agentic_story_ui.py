"""Safely migrate the existing CopilotKit story workspace to semantic agent state.

Applies to both base V1 and polished V2 UI; creates a local backup.
"""
from pathlib import Path

p = Path('frontend/src/components/StoryWorkspace.tsx')
s = p.read_text()
if 'type ToolEvent =' in s:
    print('UI semantic agent changes are already present.')
    raise SystemExit(0)
backup = p.with_suffix('.tsx.before_agentic_v2.bak')
if not backup.exists():
    backup.write_text(s)
start = s.index('type UiState = {')
end = s.index('function messageContent', start)
s = s[:start] + '''type ToolEvent = {tool: string; status: string; summary: string}
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

const FLOW = [
  { id: 'understand', label: 'Understand request', detail: 'LLM interprets the conversation and chooses a skill' },
  { id: 'classify', label: 'Classify symptoms', detail: 'DistilBERT, only when requested by the agent' },
  { id: 'recall', label: 'Check NHTSA recalls', detail: 'Only when exact vehicle context is available' },
  { id: 'retrieve', label: 'Ground recall evidence', detail: 'Vehicle-scoped FAISS + lexical retrieval' },
  { id: 'answer', label: 'Compose response', detail: 'Answer directly or with tool-verified evidence' },
]

''' + s[end:]
# Hide recall card when that tool never ran; don't imply no records.
s = s.replace('if (!category && !sources.length) return null',
              "if (!category && state.recall_status === 'not_queried') return null")
s = s.replace('      <section className="aa-result-card">\n        <div className="aa-card-eyebrow"><Database size={14}/> RECALL EVIDENCE</div>',
              '      {state.recall_status && state.recall_status !== \'not_queried\' && <section className="aa-result-card">\n        <div className="aa-card-eyebrow"><Database size={14}/> RECALL EVIDENCE</div>')
s = s.replace('No sufficiently relevant recall evidence was retrieved. This does not establish that there are no recalls or defects.',
              'No sufficiently relevant recall evidence was retrieved. This does not establish that there are no recalls or defects.')
s = s.replace('          <p className="aa-small">No sufficiently relevant recall evidence was retrieved. This does not establish that there are no recalls or defects.</p>',
              '          <p className="aa-small">{state.recall_status === \'unavailable\' ? \'NHTSA lookup unavailable; please retry or check the official VIN portal.\' : \'No sufficiently relevant recall evidence was retrieved; this does not mean no recalls exist.\'}</p>')
s = s.replace('      </section>\n    </div>\n  )\n}\n\nfunction WorkflowPanel',
              '      </section>}\n    </div>\n  )\n}\n\nfunction WorkflowPanel')
start = s.index('function WorkflowPanel(')
end = s.index('\nexport default function StoryWorkspace()', start)
s = s[:start] + '''function WorkflowPanel({ state, running, failed }: { state: UiState; running: boolean; failed: boolean }) {
  const action = state.plan?.action || ''
  const toolNames = new Set((state.tool_events || []).map(item => item.tool))
  const chosen = FLOW.filter(step =>
    step.id === 'understand' || step.id === 'answer' ||
    (step.id === 'classify' && ['classify', 'classify_and_recall'].includes(action)) ||
    (['recall', 'retrieve'].includes(step.id) && ['recall', 'classify_and_recall'].includes(action)))
  const isDone = (id: string) => id === 'understand' ? Boolean(state.plan?.action) :
    id === 'answer' ? state.stage === 'complete' :
    id === 'classify' ? toolNames.has('classify_symptoms') :
    id === 'recall' ? toolNames.has('search_recalls') : toolNames.has('retrieve_evidence')
  const status = running ? 'Working' : failed || ['agent_unavailable', 'generation_unavailable'].includes(state.stage || '') ? 'Needs attention' :
    state.stage === 'complete' ? 'Completed' : 'Ready'
  return <aside className="aa-workflow">
    <div className="aa-panel-heading"><div><div className="aa-card-eyebrow">LIVE AGENT ACTIVITY</div><h3>What the agent did</h3></div>
      <span className={running ? 'aa-running' : state.stage === 'complete' ? 'aa-finished' : 'aa-idle'}><span/>{status}</span>
    </div>
    {state.skill && <div className="aa-skill-tag"><Sparkles size={12}/> Skill: {state.skill.replaceAll('_', ' ')}</div>}
    <div className="aa-step-list">
      {chosen.map(step => {
        const done = isDone(step.id)
        const active = running && !done && chosen.filter(s => s.id !== 'answer').every(s => s.id === step.id || isDone(s.id))
        const event = (state.tool_events || []).find(e => (
          step.id === 'classify' && e.tool === 'classify_symptoms' ||
          step.id === 'recall' && e.tool === 'search_recalls' ||
          step.id === 'retrieve' && e.tool === 'retrieve_evidence'))
        return <div className={`aa-step ${done ? 'done':''} ${active?'active':''}`} key={step.id}>
          <div className="aa-step-indicator">{done ? <CheckCircle2 size={18}/> : active ? <LoaderCircle className="aa-spin" size={18}/> : <Circle size={18}/>}</div>
          <div><strong>{step.label}</strong><p>{event?.summary || step.detail}</p>
            <span className="aa-step-state">{done ? 'Completed' : active ? 'In progress' : 'Not executed'}</span>
          </div>
        </div>
      })}
    </div>
    <div className="aa-workflow-footnote">Actual LangGraph node and tool results. Planning and final answer use OpenRouter; token-level answer streaming is not enabled yet.</div>
  </aside>
}
''' + s[end:]
s = s.replace("agent.setState({ ...agent.state, stage: 'queued', classification: {}, sources: [] })",
              "agent.setState({ ...agent.state, stage: 'queued', plan: {}, classification: {}, sources: [], tool_events: [], followups: [], recall_status: 'not_queried' })")
# Include inline cards for the latest assistant message, instead of cards below the chat.
needle = '''                <div className="aa-markdown"><ReactMarkdown>{messageContent(m.content)}</ReactMarkdown></div>
              </div>
            </article>)}'''
replacement = '''                <div className="aa-markdown"><ReactMarkdown>{messageContent(m.content)}</ReactMarkdown></div>
                {m.role === 'assistant' && (() => {
                  const userTurn = messages.slice(0, idx).filter(x => x.role === 'user').length
                  const snapshot = state.result_history?.find(x => x.turn === userTurn)
                  if (!snapshot) return null
                  return <>
                    <ResultsUI state={snapshot}/>
                    {snapshot.generation_error && <p className="aa-agent-error">AI service note: {snapshot.generation_error}</p>}
                    {(snapshot.followups || []).length > 0 && <div className="aa-followups">
                      {snapshot.followups?.slice(0, 2).map((suggestion, n) => <button key={n} disabled={running} onClick={() => void send(suggestion)}>{suggestion}<ArrowRight size={13}/></button>)}
                    </div>}
                  </>
                })()}
              </div>
            </article>)}'''
if needle not in s:
    raise SystemExit('Message rendering changed; refusing unsafe rewrite.')
s = s.replace(needle, replacement, 1)
# Remove the old detached ResultsUI and fallback notice, whether polished or unpolished.
start = s.find("            {!running && state.stage === 'complete' && state.generation_mode")
if start >= 0:
    end = s.index('            <div ref={bottom}/>', start)
    s = s[:start] + s[end:]
else:
    s = s.replace("            {!running && state.stage === 'complete' && <ResultsUI state={state}/>}", '')
p.write_text(s)
print('Applied semantic agent shared-state UI patch; backup:', backup)

css = Path('frontend/src/app/globals.css')
styles = '''
/* Semantic agent tools and inline structured CopilotKit shared-state cards */
.aa-message.assistant .aa-message-content{max-width:min(92%,760px);width:100%}
.aa-message-content .aa-results{margin:17px 0 0;gap:10px}
.aa-message-content .aa-result-card{background:#f8fbfb;box-shadow:none}
.aa-skill-tag{display:flex;align-items:center;gap:7px;margin-top:15px;color:#417667;background:#f1f7f4;border:1px solid #dbeae3;padding:8px 10px;border-radius:8px;font-size:10px;text-transform:capitalize}
.aa-followups{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.aa-followups button{border:1px solid #dbe6e3;background:#fff;padding:8px 10px;border-radius:12px;color:#3c7667;font-size:10px;display:flex;align-items:center;gap:5px;text-align:left}
.aa-followups button:hover{background:#ecf6f2}
.aa-agent-error{color:#b06b37;background:#fff8ea;padding:10px;font-size:11px;border-radius:10px}
@media(max-width:660px){.aa-message.assistant .aa-message-content{max-width:93%}}
'''
contents = css.read_text()
if '/* Semantic agent tools' not in contents:
    css.write_text(contents + '\n' + styles)
