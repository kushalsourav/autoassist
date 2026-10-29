'use client'
import {useEffect, useRef} from 'react'
import {Activity, BrainCircuit, CheckCircle2, ChevronDown, Cpu, LoaderCircle, Wrench} from 'lucide-react'
import type {StreamItem} from './modelStreamEvents'

/** One live in-chat panel. No duplicate LLM answer — CopilotKit renders answer tokens. */
export default function ModelStreamViewer({items, running}: {items:StreamItem[];running:boolean}) {
  const viewport = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (viewport.current && running) viewport.current.scrollTop = viewport.current.scrollHeight
  }, [items, running])
  const reason = items.filter(e=>e.kind==='reasoning')
  const tools = items.filter(e=>e.kind==='tool-request'||e.kind==='tool-args'||e.kind==='tool-result')
  const completedTools = new Set(tools.filter(t=>t.kind==='tool-result').map(t=>t.runId))
  const models = items.filter(e=>e.kind==='model')
  const usage = items.filter(e=>e.kind==='usage')
  const chunks = reason.reduce((count,item)=>count+item.fragments,0)
  return <details className="aa-v10-live" open aria-label="Live agent model and tool activity">
    <summary>
      <span className="aa-v10-icon"><BrainCircuit size={17}/></span>
      <span className="aa-v10-heading"><strong>Live AI activity</strong><small>{models.at(-1)?.text || 'LangChain create_agent · OpenRouter'}</small></span>
      <span className={`aa-v10-status ${running?'running':''}`}>{running?<><LoaderCircle size={13} className="aa-spin"/> Streaming</>:<><Activity size={13}/> Stream ended</>}</span>
      <ChevronDown size={15}/>
    </summary>
    <div className="aa-v10-body" ref={viewport} role="log" aria-live="off">
      <div className="aa-v10-section-title"><BrainCircuit size={14}/> Provider reasoning {chunks>0&&<span>{chunks} live chunks</span>}</div>
      {reason.length ? reason.map((item,i)=><article className="aa-v10-reason" key={item.id}>
        {reason.length>1&&<div className="aa-v10-pass">Model reasoning segment {i+1}</div>}
        <pre>{item.text}</pre>
        {running && i===reason.length-1 && <span className="aa-v10-cursor" aria-hidden="true"/>}
      </article>) : <p className="aa-v10-muted">{running ? 'Waiting for provider reasoning events…' : 'No readable reasoning was emitted for this run.'}</p>}
      <div className="aa-v10-section-title"><Wrench size={14}/> Tool activity <span>{tools.filter(t=>t.kind==='tool-request').length} requested</span></div>
      {tools.length ? <div className="aa-v10-tools">{tools.map(item=><div className={`aa-v10-tool kind-${item.kind}`} key={item.id}>
        <span>{item.kind==='tool-request'?<Wrench size={13}/>:item.kind==='tool-result'?<CheckCircle2 size={13}/>:<Activity size={13}/>}</span>
        <div><strong>{item.kind==='tool-request'?`Called ${item.label}`:item.label}</strong>
          {item.kind!=='tool-request'&&<pre>{item.text}</pre>}
          {item.kind==='tool-request'&&<small>{completedTools.has(item.runId) ? 'Result received' : running ? 'Waiting for tool result' : 'No tool result observed (possibly interrupted)'}</small>}
        </div>
      </div>)}</div> : <p className="aa-v10-muted">{running?'No tool requested yet.':'The agent answered directly; no tool was called.'}</p>}
      {usage.length>0&&<div className="aa-v10-usage"><Cpu size={13}/>{usage.at(-1)?.text}</div>}
    </div>
  </details>
}
