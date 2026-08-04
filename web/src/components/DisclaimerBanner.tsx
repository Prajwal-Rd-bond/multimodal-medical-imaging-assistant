import { ShieldAlert } from 'lucide-react'

export function DisclaimerBanner() {
  return (
    <div
      className="flex items-start gap-3 rounded-xl border px-4 py-3 text-sm"
      style={{
        background: 'color-mix(in oklab, var(--status-critical) 8%, var(--surface))',
        borderColor: 'color-mix(in oklab, var(--status-critical) 35%, var(--border-strong))',
      }}
    >
      <ShieldAlert size={18} strokeWidth={1.75} style={{ color: 'var(--status-critical)' }} className="mt-0.5 shrink-0" />
      <p style={{ color: 'var(--text-secondary)' }}>
        <strong style={{ color: 'var(--text-primary)' }}>Research demo, not a diagnostic device.</strong>{' '}
        Not reviewed or approved for clinical use. Do not use this output to make or inform any
        medical decision. Always consult a qualified healthcare professional.
      </p>
    </div>
  )
}
