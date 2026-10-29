import type { Metadata } from 'next'
import type { ReactNode } from 'react'

import '@fontsource/dm-sans/400.css'
import '@fontsource/dm-sans/500.css'
import '@fontsource/dm-sans/600.css'
import '@fontsource/dm-sans/700.css'
import '@fontsource/dm-sans/800.css'

import '@fontsource/manrope/400.css'
import '@fontsource/manrope/500.css'
import '@fontsource/manrope/600.css'
import '@fontsource/manrope/700.css'
import '@fontsource/manrope/800.css'

import './globals.css'

export const metadata: Metadata = {
  title: 'AutoAssist | An Automotive AI Copilot',
  description:
    'Tell your vehicle story. Explore sourced recall evidence and the ML models behind AutoAssist.',
}

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
