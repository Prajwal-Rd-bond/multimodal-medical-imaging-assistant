import { AlertOctagon, CheckCircle2 } from 'lucide-react'

export function OodBanner({ flagged }: { flagged: boolean }) {
  if (!flagged) {
    return (
      <div
        className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm"
        style={{
          color: 'var(--status-good)',
          background: 'color-mix(in oklab, var(--status-good) 10%, transparent)',
        }}
      >
        <CheckCircle2 size={16} strokeWidth={2} />
        <span>Input looks in-distribution relative to training data.</span>
      </div>
    )
  }

  return (
    <div
      className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium"
      style={{
        color: 'var(--status-critical)',
        background: 'color-mix(in oklab, var(--status-critical) 12%, transparent)',
      }}
    >
      <AlertOctagon size={16} strokeWidth={2} />
      <span>Out-of-distribution input &mdash; treat every finding below as unreliable.</span>
    </div>
  )
}
