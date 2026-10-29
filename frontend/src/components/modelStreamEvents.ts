/** Render *observed* AG-UI stream deltas. Never read raw model prompts or headers. */
export type StreamKind = 'model' | 'reasoning' | 'tool-request' | 'tool-args' | 'tool-result' | 'usage'
export type StreamItem = {
  id: string
  seq: number
  kind: StreamKind
  label: string
  text: string
  runId: string
  fragments: number
  receivedAt: string
}
export type IncomingItem = Omit<StreamItem, 'id' | 'seq' | 'fragments' | 'receivedAt'>
type Plain = Record<string, unknown>
const obj = (v: unknown): Plain => v !== null && typeof v === 'object' && !Array.isArray(v) ? v as Plain : {}
const str = (v: unknown): string => typeof v === 'string' ? v : ''
const bound = (s: string, n = 2800) => s.slice(0, n)
const printable = (v: unknown): string => {
  if (typeof v === 'string') return bound(v)
  try { return bound(JSON.stringify(v) || '') } catch { return '[Unavailable]' }
}
const visible = (s: string): string => s.includes('[REDACTED]') ? '' : s

/** The V8 AG-UI backend emits native REASONING_MESSAGE_CONTENT and TOOL_CALL_* events.
 * AbstractAgent.subscribe callbacks usually receive {event, ...context}, not the raw event.
 * Only these native callbacks are the source of reasoning/tool UI, avoiding duplicate RAW events.
 */
export function parseAgUiStreamEvent(payload: unknown): IncomingItem[] {
  const envelope = obj(payload)
  const event = obj(envelope.event && typeof envelope.event === 'object' ? envelope.event : payload)
  const type = str(event.type)
  const id = str(event.messageId) || str(event.toolCallId) || 'ag-ui'
  const make = (kind: StreamKind, label: string, text: string): IncomingItem => ({kind, label, text, runId:id})
  if (type === 'REASONING_MESSAGE_CONTENT' || type === 'REASONING_MESSAGE_CHUNK') {
    const delta = visible(str(event.delta))
    return delta ? [make('reasoning', 'OpenRouter · provider reasoning', delta)] : []
  }
  if (type === 'TOOL_CALL_START') {
    const name = str(event.toolCallName) || str(envelope.toolCallName) || 'Tool'
    return [make('tool-request', name, 'Requested by the agent')]
  }
  if (type === 'TOOL_CALL_ARGS') {
    const delta = str(event.delta)
    return delta ? [make('tool-args', 'Tool arguments', delta.slice(0, 1200))] : []
  }
  if (type === 'TOOL_CALL_RESULT') {
    const text = printable(event.content ?? event.result)
    return [make('tool-result', 'Tool returned', text || 'Completed without output')]
  }
  return []
}

/** RAW is only used for model name + token usage; native AG-UI owns reasoning/tools.
 * In particular don't render data.input / input messages from on_chat_model_start.
 */
export function parseLangChainStream(rawPayload: unknown): IncomingItem[] {
  const envelope = obj(rawPayload)
  const raw = obj(envelope.rawEvent || (typeof envelope.event === 'object' ? envelope.event : envelope))
  const event = str(raw.event)
  const data = obj(raw.data)
  const metadata = obj(raw.metadata)
  const runId = str(raw.run_id) || 'unknown'
  if (event === 'on_chat_model_start') {
    const model = str(metadata.ls_model_name) || str(metadata.ls_provider) || 'ChatOpenRouter'
    return [{kind:'model', label:'Model request', text:model, runId}]
  }
  if (event === 'on_chat_model_end') {
    const output = obj(data.output)
    const usage = obj(output.usage_metadata)
    const details = obj(usage.output_token_details)
    const values = [
      typeof usage.input_tokens === 'number' ? `Input ${usage.input_tokens}` : '',
      typeof usage.output_tokens === 'number' ? `Output ${usage.output_tokens}` : '',
      typeof details.reasoning === 'number' ? `Reasoning ${details.reasoning}` : '',
      str(obj(output.response_metadata).finish_reason),
    ].filter(Boolean)
    return values.length ? [{kind:'usage', label:'Model usage', text:values.join(' · '), runId}] : []
  }
  return []
}

const CONTINUOUS: StreamKind[] = ['reasoning','tool-args']
/** Preserve order and distinct model passes. Coalesce token deltas, without merging tool results. */
export function appendStreamItems(previous: StreamItem[], incoming: IncomingItem[], at = new Date().toLocaleTimeString()): StreamItem[] {
  if (!incoming.length) return previous
  const result = [...previous]
  for (const item of incoming) {
    const last = result[result.length - 1]
    if (last && last.kind === item.kind && last.runId === item.runId && CONTINUOUS.includes(item.kind)) {
      result[result.length - 1] = {...last, text:(last.text + item.text).slice(-9000), fragments:last.fragments+1}
    } else {
      const seq = (last?.seq ?? 0) + 1
      result.push({...item, id:`stream-${item.runId}-${seq}`, seq, receivedAt:at, fragments:1})
    }
  }
  return result.slice(-60)
}
