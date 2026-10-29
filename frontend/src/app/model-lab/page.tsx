import Link from 'next/link'
import { Activity, ArrowLeft, BarChart3, BrainCircuit, CheckCircle2, Database, GitBranch, ShieldCheck, MessageCircle, BookOpen, Layers3, FlaskConical, ArrowRight, AlertCircle, Target, TrendingUp } from 'lucide-react'
import BrandLockup from '../../components/BrandLockup'
import { ClassifierComparisonChart, PerClassChart, RetrievalChart, RetrievalLatencyChart } from '../../components/ModelCharts'

const MODEL_RESULTS = [
  { name: 'TF-IDF + Logistic Regression', macro: .7754, micro: .8067, exact: .6738, detail: 'V3 multi-label baseline' },
  { name: 'Fine-tuned DistilBERT V3', macro: .8054, micro: .8363, exact: .7225, detail: '3 epochs · threshold 0.35' },
]

export default function ModelLab() {
  return <div className="aa-app aa-lab-shell">
    <aside className="aa-nav aa-lab-nav">
      <BrandLockup />
      <div className="aa-nav-section-label">WORKSPACE</div>
      <div className="aa-nav-items">
        <Link href="/"><MessageCircle size={17} /> Assistant</Link>
        <Link className="aa-nav-active" href="/model-lab" aria-current="page"><BookOpen size={17} /> Model Lab</Link>
      </div>
      <div className="aa-lab-nav-divider" />
      <div className="aa-nav-section-label aa-lab-nav-label">MODEL LAB</div>
      <nav className="aa-lab-toc" aria-label="Model Lab sections">
        <a href="#overview"><Layers3 size={15} /> Overview</a>
        <a href="#quality"><BarChart3 size={15} /> Model quality</a>
        <a href="#architecture"><GitBranch size={15} /> Architecture</a>
        <a href="#training"><FlaskConical size={15} /> Training details</a>
        <a href="#retrieval"><Database size={15} /> Retrieval experiments</a>
      </nav>
      <div className="aa-nav-footer"><span className="aa-footer-icon"><ShieldCheck size={17} /></span><div><strong>Evidence-first AI</strong><small>Transparent model evaluation</small></div></div>
    </aside>
    <main className="aa-main aa-lab-main">
      <header className="aa-topbar"><div className="aa-location"><span className="aa-location-icon"><BookOpen size={16} /></span><span>Workspace</span><ArrowRight size={13} /><strong>Model Lab</strong></div><Link href="/" className="aa-lab-top-link"><ArrowLeft size={15} /> Back to assistant</Link></header>
      <div className="aa-lab-body">
        <div className="aa-lab-hero" id="overview"><span className="aa-kicker">MODEL TRANSPARENCY · ENGINEERING PORTFOLIO</span><h1>Inside the <em>intelligence.</em></h1><p>A technical view of AutoAssist’s training pipeline, model quality, retrieval architecture and the limitations of our experiments.</p></div>
        <div className="aa-lab-stats">
          <div><Database size={20} /><span>Cleaned complaints</span><strong>6,052</strong><small>V3 frozen cohort</small></div>
          <div><BrainCircuit size={20} /><span>DistilBERT macro-F1</span><strong>0.8054</strong><small>Exploratory held split</small></div>
          <div><Activity size={20} /><span>Classifier mean latency</span><strong>38.0 ms</strong><small>p95 41.1 ms · warm CPU inference</small></div>
          <div><GitBranch size={20} /><span>Persistent FAISS speedup</span><strong>14.9×</strong><small>Retrieval-only · excludes network</small></div>
        </div>
        <section className="aa-lab-panel" id="quality"><div className="aa-lab-section-title"><BarChart3 size={20} /><div><h2>Model comparison</h2><p>Same V3 dataset split and taxonomy · six multi-label complaint categories</p></div></div>
          <ClassifierComparisonChart />
          <div className="aa-v6-model-facts" aria-label="Exact model comparison values">
            {MODEL_RESULTS.map(model => <div key={model.name}>
              <strong>{model.name}</strong><span>{model.detail}</span>
              <small>Macro-F1 {model.macro.toFixed(4)} · Micro-F1 {model.micro.toFixed(4)} · Exact match {model.exact.toFixed(4)}</small>
            </div>)}
          </div>
          <div className="aa-v5-gain-grid">
            <div><TrendingUp size={17} /><strong>+0.0300</strong><span>Macro-F1 improvement</span></div>
            <div><Target size={17} /><strong>+0.0296</strong><span>Micro-F1 improvement</span></div>
            <div><CheckCircle2 size={17} /><strong>+0.0487</strong><span>Exact-match improvement</span></div>
          </div>
          <p className="aa-lab-note">All differences are absolute fractions (e.g., +0.0300 = 3.00 percentage points), using the same frozen 1,211-example V3 test split. Test data was inspected during development; findings are exploratory, not independent external validation. Sigmoid scores are not calibrated probabilities.</p>
        </section>
        <div className="aa-lab-two">
          <section className="aa-lab-panel"><div className="aa-lab-section-title"><Activity size={20} /><div><h2>Per-class F1</h2><p>DistilBERT V3</p></div></div><PerClassChart /></section>
          <section className="aa-lab-panel" id="architecture"><div className="aa-lab-section-title"><GitBranch size={20} /><div><h2>Inference architecture</h2><p>From the user's story to grounded evidence</p></div></div><ol className="aa-architecture"><li>LLM interprets the question and preserves explicitly supplied vehicle context</li><li>Optional DistilBERT multi-label symptom classification</li><li>Conditional NHTSA make/model/year recall lookup</li><li>FAISS and lexical retrieval when recall evidence is requested</li><li>Stream OpenRouter answer; apply evidence and VIN-status guardrails</li></ol><div className="aa-lab-note">The graph branches according to a validated agent plan: not every question runs classification or recall tools. The V4 evidence verifier is offline-only and is not deployed.</div></section>
        </div>
        <section className="aa-lab-panel" id="training"><div className="aa-lab-section-title"><BrainCircuit size={20} /><div><h2>How the model was trained</h2><p>Reproducible experiment configuration for V3</p></div></div>
          <div className="aa-lab-experiments"><div><span>4,235 / 606 / 1,211</span><p>Training / validation / test samples (frozen split, seed 42)</p></div><div><span>3 epochs</span><p>DistilBERT; max sequence length 128; batch size 1, gradient accumulation 16</p></div><div><span>0.35</span><p>Multi-label sigmoid decision threshold, selected on validation</p></div></div>
          <p className="aa-lab-note">Classification targets are weakly derived from owner-reported NHTSA component categories, not mechanically verified failure diagnoses. Training used a local GTX 1650 GPU. A separate model was not independently externally validated.</p>
        </section>
        <section className="aa-lab-panel" id="retrieval">
          <div className="aa-lab-section-title"><ShieldCheck size={20} /><div><h2>Retrieval quality and evidence limits</h2><p>Grounded in local measurement; explicitly separated from validation claims</p></div></div>
          <div className="aa-lab-experiments">
            <div><span>67 campaigns</span><p>Unique recall campaigns in the frozen corpus · 15 successful of 17 vehicle queries</p></div>
            <div><span>204.89 → 13.79 ms</span><p>Mean warm live-versus-persistent FAISS retrieval (14.9×), one-query local benchmark, excluding NHTSA/network time</p></div>
            <div><span>18 queries / 147 judgments</span><p>Assistant-reviewed multi-vehicle relevance annotations · exploratory evaluation</p></div>
          </div>
          <RetrievalLatencyChart />
          <div className="aa-v5-retrieval-grid">
            <div className="aa-v5-retrieval-panel">
              <RetrievalChart />
              <p className="aa-v5-retrieval-footnote">Hybrid no-match false-positive rate in this inspected fixture: 22.2%. The best baseline shown here is lexical retrieval, not the deployed hybrid.</p>
            </div>
            <div className="aa-v5-retrieval-panel aa-v5-prototype-panel">
              <h3><FlaskConical size={17} /> Offline evidence-verifier experiment</h3>
              <strong>86.1% <small>Recall@3</small></strong>
              <p>Balanced reranking evaluated retrospectively on repeatedly inspected cases. This is <b>not an independently validated improvement</b> and is <b>not deployed</b>.</p>
              <div className="aa-v5-offline-label"><AlertCircle size={15} /> Do not compare this result with live-system performance.</div>
            </div>
          </div>
          <div className="aa-v5-limitations">
            <h3><AlertCircle size={17} /> What these numbers do not establish</h3>
            <p>Weak owner-reported labels are not verified diagnoses. The retrieval judgments were assistant-reviewed, the corpus covers a small vehicle sample, and inspecting evaluation cases creates selection bias. Recall listings are not proof of VIN eligibility or unrepaired status. Incomplete NHTSA results cannot prove the absence of a recall.</p>
          </div>
        </section>
      </div>
    </main>
  </div>
}
