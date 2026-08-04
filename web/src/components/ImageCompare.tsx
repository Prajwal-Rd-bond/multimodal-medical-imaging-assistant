import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { TriangleAlert } from 'lucide-react'

interface Props {
  originalUrl: string
  overlayUrl: string | null
  overlayLabel: string | null
  overlayConfident: boolean
}

export function ImageCompare({ originalUrl, overlayUrl, overlayLabel, overlayConfident }: Props) {
  const [showOverlay, setShowOverlay] = useState(true)

  return (
    <div>
      <div
        className="relative rounded-xl overflow-hidden border"
        style={{ borderColor: 'var(--border-strong)', background: '#000' }}
      >
        <AnimatePresence mode="wait">
          <motion.img
            key={showOverlay && overlayUrl ? 'overlay' : 'original'}
            src={showOverlay && overlayUrl ? overlayUrl : originalUrl}
            alt={showOverlay ? 'Grad-CAM overlay' : 'Original image'}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="w-full max-h-[420px] object-contain mx-auto"
          />
        </AnimatePresence>
      </div>

      {overlayUrl && (
        <div className="mt-3 flex items-center justify-between flex-wrap gap-2">
          <div className="inline-flex rounded-lg p-0.5" style={{ background: 'var(--surface)' }}>
            {(['overlay', 'original'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setShowOverlay(mode === 'overlay')}
                className="cursor-pointer rounded-[7px] px-3 py-1.5 text-sm font-medium transition-colors"
                style={{
                  background:
                    (mode === 'overlay') === showOverlay ? 'var(--surface-raised)' : 'transparent',
                  color: (mode === 'overlay') === showOverlay ? 'var(--text-primary)' : 'var(--text-muted)',
                  boxShadow: (mode === 'overlay') === showOverlay ? '0 1px 2px rgba(0,0,0,0.08)' : 'none',
                }}
              >
                {mode === 'overlay' ? 'Grad-CAM' : 'Original'}
              </button>
            ))}
          </div>
          {showOverlay && overlayLabel && overlayConfident && (
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              highlighting &ldquo;{overlayLabel}&rdquo;
            </span>
          )}
        </div>
      )}

      {showOverlay && overlayUrl && !overlayConfident && (
        <div
          className="mt-3 flex items-start gap-2 rounded-lg px-3 py-2.5 text-sm"
          style={{
            color: 'var(--status-warning)',
            background: 'color-mix(in oklab, var(--status-warning) 12%, transparent)',
          }}
        >
          <TriangleAlert size={16} strokeWidth={2} className="mt-0.5 shrink-0" />
          <span>
            No finding cleared the confidence threshold &mdash; this heatmap is the model&rsquo;s top
            (unconfident) guess only. Grad-CAM gradients are weak and noisy at low confidence and
            routinely highlight anatomically meaningless regions. Do not read this as a localized
            finding.
          </span>
        </div>
      )}
    </div>
  )
}
