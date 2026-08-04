export interface Finding {
  label: string
  probability: number
  calibrated_confidence: number
  flag_uncertain: boolean
}

export interface AnalyzeResponse {
  modality: 'cxr' | 'derm'
  findings: Finding[]
  ood_flag: boolean
  ood_score: number
  report_text: string
  disclaimer: string
  overlay_image: string | null
  overlay_label: string | null
  overlay_confident: boolean
  used_llm: boolean
}

export interface HealthResponse {
  status: string
  modalities: string[]
  llm_enabled: boolean
}
