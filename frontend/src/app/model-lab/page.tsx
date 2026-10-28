import Link from 'next/link'
import { Activity, ArrowLeft, BarChart3, BrainCircuit, CarFront, CheckCircle2, Database, GitBranch, ShieldCheck } from 'lucide-react'

const MODEL_RESULTS = [
  { name: 'TF-IDF + Logistic Regression', macro: .7754, micro: .8067, exact: .6738, detail: 'V3 multi-label baseline' },
  { name: 'Fine-tuned DistilBERT V3', macro: .8054, micro: .8363, exact: .7225, detail: '3 epochs · threshold 0.35' },
]
const CLASSES = [
  { name:'Airbags / Safety', f1: .8750 }, { name:'Brakes', f1: .8494 },
  { name:'Electrical', f1: .6746 }, { name:'Engine / Fuel', f1: .9061 },
  { name:'Powertrain / Drivetrain', f1: .6906 }, { name:'Steering / Suspension', f1: .8365 },
]

function Meter({ value }: { value: number }) { return <div className="aa-lab-meter"><div style={{width:`${value * 100}%`}}/></div> }

export default function ModelLab() {
  return <div className="aa-lab-page">
    <header className="aa-lab-top"><Link href="/" className="aa-lab-brand"><CarFront size={20}/> AutoAssist</Link><Link href="/" className="aa-back"><ArrowLeft size={15}/> Back to the assistant</Link></header>
    <main className="aa-lab-body">
      <div className="aa-lab-hero"><span className="aa-kicker">MODEL TRANSPARENCY · ENGINEERING PORTFOLIO</span><h1>Inside the <em>intelligence.</em></h1><p>A technical view of AutoAssist’s training pipeline, model quality, retrieval architecture and the limitations of our experiments.</p></div>
      <div className="aa-lab-stats">
        <div><Database size={20}/><span>Cleaned complaints</span><strong>6,052</strong><small>V3 frozen cohort</small></div>
        <div><BrainCircuit size={20}/><span>DistilBERT macro-F1</span><strong>0.8054</strong><small>Exploratory held split</small></div>
        <div><Activity size={20}/><span>Classifier mean latency</span><strong>38.0 ms</strong><small>CPU only · 20 warm requests</small></div>
        <div><GitBranch size={20}/><span>Persistent FAISS speedup</span><strong>14.9×</strong><small>One-query warm comparison</small></div>
      </div>
      <section className="aa-lab-panel"><div className="aa-lab-section-title"><BarChart3 size={20}/><div><h2>Model comparison</h2><p>Same V3 dataset split and taxonomy · six multi-label complaint categories</p></div></div>
        <div className="aa-model-grid">{MODEL_RESULTS.map(m => <div className="aa-lab-model" key={m.name}><div className="aa-lab-model-head"><h3>{m.name}</h3><span>{m.detail}</span></div>
          {[['Macro-F1',m.macro],['Micro-F1',m.micro],['Exact match',m.exact]].map(([name,value]) => <div className="aa-model-metric" key={String(name)}><div><span>{name}</span><b>{Number(value).toFixed(4)}</b></div><Meter value={Number(value)}/></div>)}
        </div>)}</div><p className="aa-lab-note">Test data was inspected during development; results are exploratory and not external validation. Sigmoid output scores are not calibrated probabilities.</p>
      </section>
      <div className="aa-lab-two">
        <section className="aa-lab-panel"><div className="aa-lab-section-title"><Activity size={20}/><div><h2>Per-class F1</h2><p>DistilBERT V3</p></div></div>{CLASSES.map(c => <div className="aa-model-metric" key={c.name}><div><span>{c.name}</span><b>{c.f1.toFixed(3)}</b></div><Meter value={c.f1}/></div>)}</section>
        <section className="aa-lab-panel"><div className="aa-lab-section-title"><GitBranch size={20}/><div><h2>Inference architecture</h2><p>From the user's story to grounded evidence</p></div></div><ol className="aa-architecture"><li>Identify vehicle through chat</li><li>DistilBERT multi-label component classification</li><li>Vehicle-specific NHTSA live recall lookup</li><li>Persistent FAISS + lexical relevance retrieval</li><li>OpenRouter LLM generation with VIN caveats</li></ol><div className="aa-lab-note">The V4 evidence verifier is offline-only; it has not replaced the running retriever.</div></section>
      </div>
      <section className="aa-lab-panel"><div className="aa-lab-section-title"><BrainCircuit size={20}/><div><h2>How the model was trained</h2><p>Reproducible experiment configuration for V3</p></div></div>
        <div className="aa-lab-experiments"><div><span>4,235 / 606 / 1,211</span><p>Training / validation / test samples (frozen split, seed 42)</p></div><div><span>3 epochs</span><p>DistilBERT; max sequence length 128; batch size 1, gradient accumulation 16</p></div><div><span>0.35</span><p>Multi-label sigmoid decision threshold, selected on validation</p></div></div>
        <p className="aa-lab-note">Classification targets are weakly derived from owner-reported NHTSA component categories, not mechanically verified failure diagnoses. Training used a local GTX 1650 GPU. A separate model was not independently externally validated.</p>
      </section>
      <section className="aa-lab-panel"><div className="aa-lab-section-title"><ShieldCheck size={20}/><div><h2>Retrieval experiments and safeguards</h2><p>Separate pilot findings from measured production behavior</p></div></div><div className="aa-lab-experiments"><div><span>67</span><p>Unique campaigns indexed across 15 successful vehicle queries</p></div><div><span>13.79 ms</span><p>Mean warm persistent retrieval on one Honda query (network excluded)</p></div><div><span>86.1%</span><p>V4 balanced Recall@3 on repeatedly inspected 18-query fixture only</p></div></div><p className="aa-lab-note"><CheckCircle2 size={14}/> Recall listings do not confirm VIN eligibility, an unrepaired campaign, or a mechanical diagnosis. Incomplete NHTSA fetches cannot be treated as no recalls.</p></section>
    </main>
  </div>
}
