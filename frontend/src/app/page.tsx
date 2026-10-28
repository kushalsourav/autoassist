'use client'

import { CopilotKitProvider } from '@copilotkit/react-core/v2'
import StoryWorkspace from '../components/StoryWorkspace'

export default function Home() {
  return (
    <CopilotKitProvider
      runtimeUrl="/api/copilotkit"
      agentId="autoassist"
    >
      <StoryWorkspace />
    </CopilotKitProvider>
  )
}
