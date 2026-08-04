import { Bone, Scan } from 'lucide-react'

const CONFIG = {
  cxr: { label: 'Chest X-ray', icon: Bone, color: 'var(--brand)' },
  derm: { label: 'Dermoscopy', icon: Scan, color: 'var(--cat-derm)' },
} as const

export function ModalityBadge({ modality }: { modality: 'cxr' | 'derm' }) {
  const { label, icon: Icon, color } = CONFIG[modality]
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium"
      style={{ background: `color-mix(in oklab, ${color} 14%, transparent)`, color }}
    >
      <Icon size={14} strokeWidth={2} />
      {label}
    </span>
  )
}
