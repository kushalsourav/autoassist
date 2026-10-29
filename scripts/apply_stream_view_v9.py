"""Small, idempotent V9 frontend installer. Requires V8; preserves all V4/V7 CSS."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / 'frontend/src/components/StoryWorkspace.tsx'
CSS = ROOT / 'frontend/src/app/globals.css'
STYLE = (Path(__file__).parent / 'assets/autoassist-v9-stream.css').read_text(encoding='utf-8')


def replace_once(source: str, before: str, after: str, label: str) -> str:
    if after in source:
        return source
    if source.count(before) != 1:
        raise RuntimeError(f'V9 compatibility check failed: {label} (expected 1 match, found {source.count(before)}). Confirm V8 is installed.')
    return source.replace(before, after, 1)


def apply():
    if not FRONT.exists() or not CSS.exists():
        raise FileNotFoundError('Run at the AutoAssist repo root after installing V8')
    source = FRONT.read_text(encoding='utf-8')
    source = replace_once(source,
        "import BrandLockup from './BrandLockup'",
        "import BrandLockup from './BrandLockup'\nimport ModelStreamViewer from './ModelStreamViewer'\nimport { appendStreamItems, parseLangChainStream, type StreamItem } from './modelStreamEvents'",
        'stream viewer import')
    source = replace_once(source,
        "const SHOW_PROVIDER_REASONING = process.env.NEXT_PUBLIC_AUTOASSIST_PROVIDER_REASONING === 'true'",
        "const SHOW_PROVIDER_REASONING = process.env.NEXT_PUBLIC_AUTOASSIST_PROVIDER_REASONING === 'true'\nconst SHOW_STREAM_DEBUG = process.env.NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG === 'true'",
        'dev mode env')
    source = replace_once(source,
        "  const [providerReasoning, setProviderReasoning] = useState('')",
        "  const [providerReasoning, setProviderReasoning] = useState('')\n  const [streamEvents, setStreamEvents] = useState<StreamItem[]>([])",
        'stream events state')
    source = replace_once(source,
        "      onRawEvent: (payload: unknown) => {\n        if (!SHOW_PROVIDER_REASONING) return",
        "      onRawEvent: (payload: unknown) => {\n        if (SHOW_STREAM_DEBUG) {\n          const entries = parseLangChainStream(payload)\n          if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))\n        }\n        if (!SHOW_PROVIDER_REASONING) return",
        'live raw subscriptions')
    # On thread switches, reset events so no unrelated conversation is mixed in.
    source = replace_once(source,
        "setStepsByTurn({}); setActiveThread(id); setLiveSteps([]); setStreamedText(''); setProviderReasoning(''); setError('')",
        "setStepsByTurn({}); setActiveThread(id); setLiveSteps([]); setStreamedText(''); setProviderReasoning(''); setStreamEvents([]); setError('')",
        'new thread reset')
    source = replace_once(source,
        "setStepsByTurn(stepsByTurnRef.current); setActiveThread(entry.id); setLiveSteps([]); setStreamedText(''); setProviderReasoning(''); setError('')",
        "setStepsByTurn(stepsByTurnRef.current); setActiveThread(entry.id); setLiveSteps([]); setStreamedText(''); setProviderReasoning(''); setStreamEvents([]); setError('')",
        'open thread reset')
    source = replace_once(source,
        "setPrompt(''); setStreamedText(''); setProviderReasoning(''); setLiveSteps([])",
        "setPrompt(''); setStreamedText(''); setProviderReasoning(''); setStreamEvents([]); setLiveSteps([])",
        'send reset')
    source = replace_once(source,
        '            {providerReasoning && <details className="aa-provider-reasoning"',
        '            {!SHOW_STREAM_DEBUG && providerReasoning && <details className="aa-provider-reasoning"',
        'deduplicate provider panel')
    source = replace_once(source,
        '        <div className="aa-side-rail">\n          <GuidePanel/>',
        '        <div className="aa-side-rail">\n          {SHOW_STREAM_DEBUG && <ModelStreamViewer items={streamEvents} running={running}/>}\n          <GuidePanel/>',
        'developer sidebar')
    FRONT.write_text(source, encoding='utf-8')
    css_text = CSS.read_text(encoding='utf-8')
    if '/* V9 — opt-in developer stream inspector' not in css_text:
        CSS.write_text(css_text.rstrip() + '\n\n' + STYLE + '\n', encoding='utf-8')
    print('V9 stream viewer installed (safe to run again).')


if __name__ == '__main__':
    apply()
