import { motion } from 'framer-motion'
import { TriangleAlert } from 'lucide-react'
import type { Finding } from '../lib/types'

interface Props {
  finding: Finding
  delay?: number
}

export function ConfidenceBar({ finding, delay = 0 }: Props) {
  const pct = Math.round(finding.calibrated_confidence * 100)

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="font-medium text-sm truncate" style={{ color: 'var(--text-primary)' }}>
            {finding.label}
          </span>
          {finding.flag_uncertain && (
            <span
              className="flex items-center gap-1 shrink-0 rounded-full px-1.5 py-0.5 text-[11px] font-medium"
              style={{
                color: 'var(--status-warning)',
                background: 'color-mix(in oklab, var(--status-warning) 14%, transparent)',
              }}
            >
              <TriangleAlert size={11} strokeWidth={2} />
              uncertain
            </span>
          )}
        </div>
        <span
          className="text-sm shrink-0 pl-2"
          style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}
        >
          {pct}%
        </span>
      </div>
      <div
        className="relative w-full rounded-full overflow-hidden"
        style={{ height: 8, background: 'var(--seq-100)' }}
      >
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, delay, ease: 'easeOut' }}
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: 'var(--seq-400)' }}
        />
      </div>
    </div>
  )
}
