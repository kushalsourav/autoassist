'use client'

/** Portfolio charts, styled in the Google Gemini / Material Design 3 visual language and powered by Recharts.
 * All values are measured offline fixtures, NOT live telemetry or independently held-out estimates.
 */
import type { ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const COLORS = {
  base: '#94a3b8',      // Slate baseline
  tuned: '#4f6bf5',     // Gemini vibrant blue
  teal: '#00a884',      // Gemini teal / emerald
  lexical: '#7c5ce6',   // Gemini purple / iris
}

const quality = [
  { metric: 'Macro-F1', baseline: 77.54, transformer: 80.54 },
  { metric: 'Micro-F1', baseline: 80.67, transformer: 83.63 },
  { metric: 'Exact match', baseline: 67.38, transformer: 72.25 },
]

const components = [
  { name: 'Airbags / Safety', score: 87.50 },
  { name: 'Brakes', score: 84.94 },
  { name: 'Electrical', score: 67.46 },
  { name: 'Engine / Fuel', score: 90.61 },
  { name: 'Powertrain', score: 69.06 },
  { name: 'Steering / Suspension', score: 83.65 },
]

const retrieval = [
  { method: 'Semantic', recall: 22.2 },
  { method: 'TF-IDF lexical', recall: 61.1 },
  { method: 'Hybrid deployed', recall: 27.8 },
]

const latency = [
  { method: 'Live FAISS path', ms: 204.89 },
  { method: 'Persisted index', ms: 13.79 },
]

const axisStyle = { fontSize: 11, fill: '#64748b', fontWeight: 500 }

const tooltipStyle = {
  backgroundColor: 'rgba(255, 255, 255, 0.98)',
  borderRadius: 12,
  border: '1px solid #e2e8f0',
  boxShadow: '0 8px 24px rgba(15, 23, 42, 0.08)',
  fontSize: 12,
  padding: '8px 12px',
}

function ChartShell({ children, title, subtitle, height = 260 }: { children: ReactNode; title: string; subtitle: string; height?: number }) {
  return (
    <div className="aa-v6-chart" role="group" aria-label={title}>
      <div className="aa-v6-chart-head">
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
      <div style={{ height, width: '100%', minWidth: 0 }}>{children}</div>
    </div>
  )
}

const pctLabel = (value: unknown) => `${Number(value).toFixed(1)}%`

export function ClassifierComparisonChart() {
  return (
    <ChartShell title="Model quality at a glance" subtitle="Same frozen V3 split · percentages · larger is better" height={280}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={quality} margin={{ top: 12, right: 16, left: -6, bottom: 4 }} barGap={8} accessibilityLayer>
          <CartesianGrid vertical={false} stroke="#edf2f7" strokeDasharray="3 3" />
          <XAxis dataKey="metric" tick={axisStyle} axisLine={false} tickLine={false} tickMargin={12} />
          <YAxis domain={[0, 100]} tick={axisStyle} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} width={40} />
          <Tooltip formatter={pctLabel} contentStyle={tooltipStyle} cursor={{ fill: '#f1f5f9' }} />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 12 }} />
          <Bar name="TF-IDF baseline" dataKey="baseline" fill={COLORS.base} radius={[6, 6, 0, 0]} maxBarSize={44} isAnimationActive={false} />
          <Bar name="DistilBERT V3" dataKey="transformer" fill={COLORS.tuned} radius={[6, 6, 0, 0]} maxBarSize={44} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  )
}

export function PerClassChart() {
  return (
    <ChartShell title="Per-component F1" subtitle="Weak component labels · DistilBERT V3 · percentages" height={320}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={components} layout="vertical" margin={{ top: 4, right: 20, left: 4, bottom: 4 }} accessibilityLayer>
          <CartesianGrid horizontal={false} stroke="#edf2f7" strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} tick={axisStyle} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={130} tick={{ ...axisStyle, fontSize: 10.5 }} axisLine={false} tickLine={false} />
          <Tooltip formatter={pctLabel} contentStyle={tooltipStyle} cursor={{ fill: '#f1f5f9' }} />
          <Bar name="F1" dataKey="score" fill={COLORS.teal} radius={[0, 6, 6, 0]} maxBarSize={22} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  )
}

export function RetrievalChart() {
  return (
    <ChartShell title="Recall@3 — inspected fixture" subtitle="18 queries / 147 assistant-reviewed judgments · NOT a held-out benchmark" height={250}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={retrieval} margin={{ top: 12, right: 12, left: -10, bottom: 6 }} accessibilityLayer>
          <CartesianGrid vertical={false} stroke="#edf2f7" strokeDasharray="3 3" />
          <XAxis dataKey="method" tick={{ ...axisStyle, fontSize: 10.5 }} axisLine={false} tickLine={false} tickMargin={10} />
          <YAxis domain={[0, 100]} tick={axisStyle} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} width={40} />
          <Tooltip formatter={pctLabel} contentStyle={tooltipStyle} cursor={{ fill: '#f1f5f9' }} />
          <Bar name="Recall@3" dataKey="recall" fill={COLORS.lexical} radius={[6, 6, 0, 0]} maxBarSize={56} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  )
}

export function RetrievalLatencyChart() {
  return (
    <ChartShell title="Warm retrieval latency" subtitle="Local, one-query retrieval-only comparison · milliseconds · lower is better" height={195}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={latency} layout="vertical" margin={{ top: 2, right: 24, left: 4, bottom: 4 }} accessibilityLayer>
          <CartesianGrid horizontal={false} stroke="#edf2f7" strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 230]} tick={axisStyle} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="method" width={130} tick={{ ...axisStyle, fontSize: 10.5 }} axisLine={false} tickLine={false} />
          <Tooltip formatter={value => `${Number(value).toFixed(2)} ms`} contentStyle={tooltipStyle} cursor={{ fill: '#f1f5f9' }} />
          <Bar name="Mean latency" dataKey="ms" fill={COLORS.teal} radius={[0, 6, 6, 0]} maxBarSize={24} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  )
}
