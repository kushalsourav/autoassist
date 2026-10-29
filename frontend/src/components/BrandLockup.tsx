import Link from 'next/link'
import { CarFront } from 'lucide-react'

/** Shared navigation identity across chat and Model Lab. */
export default function BrandLockup() {
  return (
    <Link href="/" className="aa-brand" aria-label="AutoAssist home">
      <span className="aa-brand-mark"><CarFront size={20} strokeWidth={2.1}/></span>
      <span className="aa-brand-copy">autoassist<span className="aa-brand-sub">AUTOMOTIVE INTELLIGENCE</span></span>
    </Link>
  )
}
