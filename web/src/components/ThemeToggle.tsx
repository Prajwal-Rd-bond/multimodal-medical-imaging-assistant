import { useEffect, useState } from 'react'
import { Moon, Sun } from 'lucide-react'

type Theme = 'light' | 'dark'

function getInitialTheme(): Theme {
  const stored = localStorage.getItem('theme')
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  return (
    <button
      onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
      aria-label="Toggle color theme"
      className="cursor-pointer flex items-center justify-center h-9 w-9 rounded-full border transition-colors hover:bg-(--surface-raised)"
      style={{ borderColor: 'var(--border)' }}
    >
      {theme === 'light' ? (
        <Moon size={16} strokeWidth={1.75} style={{ color: 'var(--text-secondary)' }} />
      ) : (
        <Sun size={16} strokeWidth={1.75} style={{ color: 'var(--text-secondary)' }} />
      )}
    </button>
  )
}
