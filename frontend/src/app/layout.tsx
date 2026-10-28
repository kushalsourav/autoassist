import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import './globals.css'

export const metadata: Metadata = {
  title: 'AutoAssist | An Automotive AI Copilot',
  description: 'Tell your vehicle story. Explore sourced recall evidence and the ML models behind AutoAssist.',
}

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>
}
