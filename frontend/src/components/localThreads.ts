/** Local, single-browser development history. Not a server-side secure store. */
export type SavedThread = {
  id: string
  title: string
  updatedAt: number
  messages: unknown[]
  state: Record<string, unknown>
  // Per-turn client-observed AG-UI node events (optional for older threads).
  stepsByTurn?: Record<string, Array<{ name: string; status: 'running' | 'complete'; started: number; durationMs?: number }>>
}

const KEY = 'autoassist.dev.threads.v1'

export function readLocalThreads(): SavedThread[] {
  if (typeof window === 'undefined') return []
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(KEY) || '[]')
    if (!Array.isArray(value)) return []
    return value.filter((t): t is SavedThread =>
      !!t && typeof t === 'object' && typeof t.id === 'string' &&
      typeof t.title === 'string' && Array.isArray(t.messages) &&
      typeof t.state === 'object' && t.state !== null &&
      typeof t.updatedAt === 'number').slice(0, 30)
  } catch { return [] }
}

export function writeLocalThreads(threads: SavedThread[]): void {
  if (typeof window === 'undefined') return
  try { window.localStorage.setItem(KEY, JSON.stringify(threads.slice(0, 30))) }
  catch { /* Storage disabled / exceeded quota; UI still works without history. */ }
}
