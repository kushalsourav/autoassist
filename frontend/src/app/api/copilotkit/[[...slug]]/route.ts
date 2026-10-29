import { HttpAgent } from '@ag-ui/client'
import { CopilotRuntime, InMemoryAgentRunner, createCopilotRuntimeHandler } from '@copilotkit/runtime/v2'

// Server-side URL. Never use NEXT_PUBLIC_* for agent secrets or internal credentials.
const agentUrl = process.env.AUTOASSIST_AGUI_URL || 'http://127.0.0.1:8000/agent'
const runtime = new CopilotRuntime({
  agents: { autoassist: new HttpAgent({ url: agentUrl, ...(process.env.AUTOASSIST_AGENT_TOKEN ? { headers: { 'x-autoassist-agent-token': process.env.AUTOASSIST_AGENT_TOKEN } } : {}) }) },
  runner: new InMemoryAgentRunner(),
})
const handler = createCopilotRuntimeHandler({ runtime, basePath: '/api/copilotkit' })
export const GET = handler
export const POST = handler
export const PATCH = handler
export const DELETE = handler
