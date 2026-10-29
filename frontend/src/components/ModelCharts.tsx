'use client'

/** Portfolio charts, styled in the shadcn visual language and powered by Recharts.
 * All values are measured offline fixtures, NOT live telemetry or independently held-out estimates.
 */
import type { ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const COLORS = { base: '#97a2ba', tuned: '#5966c4', teal: '#32968d', lexical: '#656dc3' }

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

const axis = { fontSize: 11, fill: '#71809b' }

function ChartShell({ children, title, subtitle, height = 260 }: { children: ReactNode; title: string; subtitle: string; height?: number }) {
  return <div className="aa-v6-chart" role="group" aria-label={title}>
    <div className="aa-v6-chart-head"><h3>{title}</h3><p>{subtitle}</p></div>
    <div style={{height, width: '100%', minWidth: 0}}>{children}</div>
  </div>
}

const pctLabel = (value: unknown) => `${Number(value).toFixed(1)}%`

export function ClassifierComparisonChart() {
  return <ChartShell title="Model quality at a glance" subtitle="Same frozen V3 split · percentages · larger is better" height={274}>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={quality} margin={{top:10,right:12,left:-8,bottom:2}} barGap={6} accessibilityLayer>
        <CartesianGrid vertical={false} stroke="#e7eaf4" strokeDasharray="3 3"/>
        <XAxis dataKey="metric" tick={axis} axisLine={false} tickLine={false} tickMargin={12}/>
        <YAxis domain={[0,100]} tick={axis} tickFormatter={v=>`${v}%`} axisLine={false} tickLine={false}/>
        <Tooltip formatter={pctLabel} contentStyle={{borderRadius:14,border:'1px solid #e0e5f2',fontSize:12}}/>
        <Legend wrapperStyle={{fontSize:11,paddingTop:10}}/>
        <Bar name="TF-IDF baseline" dataKey="baseline" fill={COLORS.base} radius={[6,6,0,0]} maxBarSize={42} isAnimationActive={false}/>
        <Bar name="DistilBERT V3" dataKey="transformer" fill={COLORS.tuned} radius={[6,6,0,0]} maxBarSize={42} isAnimationActive={false}/>
      </BarChart>
    </ResponsiveContainer>
  </ChartShell>
}

export function PerClassChart() {
  return <ChartShell title="Per-component F1" subtitle="Weak component labels · DistilBERT V3 · percentages" height={316}>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={components} layout="vertical" margin={{top:0,right:15,left:0,bottom:0}} accessibilityLayer>
        <CartesianGrid horizontal={false} stroke="#e7eaf4" strokeDasharray="3 3"/>
        <XAxis type="number" domain={[0,100]} tick={axis} tickFormatter={v=>`${v}%`} axisLine={false} tickLine={false}/>
        <YAxis type="category" dataKey="name" width={126} tick={{...axis,fontSize:10}} axisLine={false} tickLine={false}/>
        <Tooltip formatter={pctLabel} contentStyle={{borderRadius:14,border:'1px solid #e0e5f2',fontSize:12}}/>
        <Bar name="F1" dataKey="score" fill={COLORS.teal} radius={[0,6,6,0]} maxBarSize={20} isAnimationActive={false}/>
      </BarChart>
    </ResponsiveContainer>
  </ChartShell>
}

export function RetrievalChart() {
  return <ChartShell title="Recall@3 — inspected fixture" subtitle="18 queries / 147 assistant-reviewed judgments · NOT a held-out benchmark" height={245}>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={retrieval} margin={{top:12,right:6,left:-12,bottom:4}} accessibilityLayer>
        <CartesianGrid vertical={false} stroke="#e7eaf4" strokeDasharray="3 3"/>
        <XAxis dataKey="method" tick={{...axis,fontSize:10}} axisLine={false} tickLine={false} tickMargin={10}/>
        <YAxis domain={[0,100]} tick={axis} tickFormatter={v=>`${v}%`} axisLine={false} tickLine={false}/>
        <Tooltip formatter={pctLabel} contentStyle={{borderRadius:14,border:'1px solid #e0e5f2',fontSize:12}}/>
        <Bar name="Recall@3" dataKey="recall" fill={COLORS.lexical} radius={[6,6,0,0]} maxBarSize={65} isAnimationActive={false}/>
      </BarChart>
    </ResponsiveContainer>
  </ChartShell>
}

export function RetrievalLatencyChart() {
  return <ChartShell title="Warm retrieval latency" subtitle="Local, one-query retrieval-only comparison · milliseconds · lower is better" height={190}>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={latency} layout="vertical" margin={{top:0,right:20,left:0,bottom:2}} accessibilityLayer>
        <CartesianGrid horizontal={false} stroke="#e7eaf4" strokeDasharray="3 3"/>
        <XAxis type="number" domain={[0,230]} tick={axis} axisLine={false} tickLine={false}/>
        <YAxis type="category" dataKey="method" width={123} tick={{...axis,fontSize:10}} axisLine={false} tickLine={false}/>
        <Tooltip formatter={value=>`${Number(value).toFixed(2)} ms`} contentStyle={{borderRadius:14,border:'1px solid #e0e5f2',fontSize:12}}/>
        <Bar name="Mean latency" dataKey="ms" fill={COLORS.teal} radius={[0,6,6,0]} maxBarSize={23} isAnimationActive={false}/>
      </BarChart>
    </ResponsiveContainer>
  </ChartShell>
}
