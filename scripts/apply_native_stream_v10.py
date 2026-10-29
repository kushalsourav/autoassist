"""V10: fix V9 frontend to consume native AG-UI events live in conversation. Requires V9."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / 'frontend/src/components/StoryWorkspace.tsx'
CSS = ROOT / 'frontend/src/app/globals.css'
STYLE = (Path(__file__).parent / 'assets/autoassist-v10-live.css').read_text(encoding='utf8')


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    occurrences = text.count(old)
    if occurrences != 1:
        raise RuntimeError(f'V10 patch: {label}: expected exactly one V9 match, got {occurrences}. Make sure V9 was applied.')
    return text.replace(old, new, 1)


def apply() -> None:
    if not FRONT.is_file():
        raise FileNotFoundError('Missing frontend/src/components/StoryWorkspace.tsx; run installer from root after V9')
    text = FRONT.read_text(encoding='utf8')
    text = replace_once(text,
        'import { appendStreamItems, parseLangChainStream, type StreamItem } from \'./modelStreamEvents\'',
        'import { appendStreamItems, parseAgUiStreamEvent, parseLangChainStream, type StreamItem } from \'./modelStreamEvents\'',
        'AG-UI importer')
    text = replace_once(text,
        "const SHOW_STREAM_DEBUG = process.env.NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG === 'true'",
        "const SHOW_STREAM_DEBUG = process.env.NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG === 'true' ||\n  (process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG !== 'false')",
        'developer viewer default')
    text = replace_once(text,
        '      onStepStartedEvent: (event: unknown) => {',
        '''      // The backend emits *native* AG-UI reasoning + tool events alongside RAW.
      // Subscribe here rather than relying on RAW decoding or an env-only panel.
      onReasoningMessageContentEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onToolCallStartEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onToolCallArgsEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onToolCallResultEvent: (payload: unknown) => {
        if (!SHOW_STREAM_DEBUG) return
        const entries = parseAgUiStreamEvent(payload)
        if (entries.length) setStreamEvents(previous => appendStreamItems(previous, entries))
      },
      onStepStartedEvent: (event: unknown) => {''',
        'native event callbacks')
    text = replace_once(text,
        '            {running && <LiveActivity steps={liveSteps}/>} ',
        '            {running && !SHOW_STREAM_DEBUG && <LiveActivity steps={liveSteps}/>} ',
        'prevent parallel step UI') if '            {running && <LiveActivity steps={liveSteps}/>} ' in text else replace_once(text,
        '            {running && <LiveActivity steps={liveSteps}/>}',
        '            {running && !SHOW_STREAM_DEBUG && <LiveActivity steps={liveSteps}/>}',
        'prevent parallel step UI')
    text = replace_once(text,
        '            {!SHOW_STREAM_DEBUG && providerReasoning && <details',
        '            {SHOW_STREAM_DEBUG && (running || streamEvents.length > 0) && <ModelStreamViewer items={streamEvents} running={running}/>}\n            {!SHOW_STREAM_DEBUG && providerReasoning && <details',
        'in-chat live panel')
    # Deletion needs an explicit test: the short replacement is always a substring
    # of the original, so replace_once's idempotence shortcut is not suitable.
    old_side = '          {SHOW_STREAM_DEBUG && <ModelStreamViewer items={streamEvents} running={running}/>}\n          <GuidePanel/>'
    if old_side in text:
        text = text.replace(old_side, '          <GuidePanel/>', 1)
    FRONT.write_text(text, encoding='utf8')
    css = CSS.read_text(encoding='utf8')
    if '/* V10: One in-chat live AG-UI reasoning' not in css:
        CSS.write_text(css.rstrip() + '\n\n' + STYLE + '\n', encoding='utf8')
    print('V10 installed: live AG-UI reasoning/tools appear in conversation during execution.')

if __name__ == '__main__':
    apply()
