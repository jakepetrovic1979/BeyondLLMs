import { prisma } from '../db'
import { Analytics } from '@prisma/client'

export interface MetaState {
  predictionError: number
  complexity: number
  entropy: number
  novelty: number
  phi: number
  opacity: number
  bindingStrength: number
  taskDifficulty: number
  distributionShift: number
  anomalyScore: number
}

export interface Improvement {
  type: string
  description: string
  code: string
  riskLevel: 'low' | 'medium' | 'high'
}

/**
 * Compute meta-states from analytics data.
 * This is the "sensing" part of System M.
 */
export async function computeMetaState(
  appId: string,
  analytics: Analytics[]
): Promise<MetaState> {
  const predictionError = computePredictionError(analytics)
  const phi = computePhiIntegration(analytics)
  const taskDifficulty = computeTaskDifficulty(analytics)

  return {
    predictionError,
    complexity: estimateComplexity(analytics),
    entropy: computeActionEntropy(analytics),
    novelty: computeNovelty(analytics),
    phi,
    opacity: computeOpacity(analytics),
    bindingStrength: computeBindingStrength(analytics),
    taskDifficulty,
    distributionShift: await detectDistributionShift(appId, analytics),
    anomalyScore: detectAnomalies(analytics),
  }
}

function computePredictionError(analytics: Analytics[]): number {
  if (analytics.length === 0) return 0.5

  const actionCounts: Record<string, number> = {}
  for (const a of analytics) {
    actionCounts[a.actionType] = (actionCounts[a.actionType] || 0) + 1
  }

  const values = Object.values(actionCounts)
  const actualEntropy = shannonEntropy(values)
  const expectedEntropy = Math.log(Math.max(Object.keys(actionCounts).length, 1))

  if (expectedEntropy === 0) return 0
  return Math.min(1.0, Math.abs(actualEntropy - expectedEntropy) / expectedEntropy)
}

function computePhiIntegration(analytics: Analytics[]): number {
  if (analytics.length < 2) return 0.5

  // Measure sequence coherence: how predictable is the next action?
  const transitions: Record<string, Record<string, number>> = {}
  for (let i = 1; i < analytics.length; i++) {
    const prev = analytics[i - 1].actionType
    const curr = analytics[i].actionType
    if (!transitions[prev]) transitions[prev] = {}
    transitions[prev][curr] = (transitions[prev][curr] || 0) + 1
  }

  // High coherence = dominant transitions exist
  let totalCoherence = 0
  let count = 0
  for (const prev of Object.keys(transitions)) {
    const nextCounts = Object.values(transitions[prev])
    const total = nextCounts.reduce((a, b) => a + b, 0)
    const max = Math.max(...nextCounts)
    totalCoherence += max / total
    count++
  }

  return count > 0 ? totalCoherence / count : 0.5
}

function computeTaskDifficulty(analytics: Analytics[]): number {
  if (analytics.length === 0) return 0.3

  const errorCount = analytics.filter((a) => a.errorMessage).length
  const uniqueActions = new Set(analytics.map((a) => a.actionType)).size

  return Math.min(1.0, (uniqueActions * 0.05 + (errorCount / analytics.length) * 0.5))
}

function estimateComplexity(analytics: Analytics[]): number {
  const uniqueActions = new Set(analytics.map((a) => a.actionType)).size
  return Math.min(1.0, uniqueActions / 20)
}

function computeActionEntropy(analytics: Analytics[]): number {
  if (analytics.length === 0) return 0.5
  const counts: Record<string, number> = {}
  for (const a of analytics) {
    counts[a.actionType] = (counts[a.actionType] || 0) + 1
  }
  const maxEntropy = Math.log(Math.max(Object.keys(counts).length, 1))
  if (maxEntropy === 0) return 0
  return shannonEntropy(Object.values(counts)) / maxEntropy
}

function computeNovelty(analytics: Analytics[]): number {
  if (analytics.length < 10) return 0.5
  const recent = analytics.slice(0, 10)
  const older = analytics.slice(10)
  const recentTypes = new Set(recent.map((a) => a.actionType))
  const olderTypes = new Set(older.map((a) => a.actionType))
  let novelCount = 0
  for (const t of recentTypes) {
    if (!olderTypes.has(t)) novelCount++
  }
  return novelCount / Math.max(recentTypes.size, 1)
}

function computeOpacity(analytics: Analytics[]): number {
  if (analytics.length === 0) return 0.5
  return analytics.filter((a) => a.errorMessage).length / analytics.length
}

function computeBindingStrength(analytics: Analytics[]): number {
  // Proxy: ratio of sequential vs random-looking navigation
  return computePhiIntegration(analytics)
}

async function detectDistributionShift(
  _appId: string,
  analytics: Analytics[]
): Promise<number> {
  const recent = analytics.slice(0, 100)
  const historical = analytics.slice(100, 200)

  if (historical.length === 0) return 0

  const recentDist = getActionDistribution(recent)
  const histDist = getActionDistribution(historical)

  return klDivergence(recentDist, histDist)
}

function detectAnomalies(analytics: Analytics[]): number {
  if (analytics.length < 10) return 0

  // Check error spike in recent actions
  const recent = analytics.slice(0, 10)
  const recentErrorRate = recent.filter((a) => a.errorMessage).length / recent.length
  const overallErrorRate = analytics.filter((a) => a.errorMessage).length / analytics.length

  return Math.min(1.0, Math.max(0, recentErrorRate - overallErrorRate) * 5)
}

/**
 * System M policy: decide what improvements to make based on meta-states.
 */
export async function decideImprovements(metaState: MetaState): Promise<Improvement[]> {
  const improvements: Improvement[] = []

  if (metaState.predictionError > 0.4) {
    improvements.push({
      type: 'collect_feedback',
      description:
        'App behavior does not match user expectations. Collecting feedback to understand the gap.',
      code: '// openFeedbackForm()',
      riskLevel: 'low',
    })
  }

  if (metaState.phi < 0.4) {
    improvements.push({
      type: 'refactor_components',
      description:
        'Components are weakly integrated. Restructuring component hierarchy for better coherence.',
      code: '// Restructure component hierarchy for improved integration',
      riskLevel: 'high',
    })
  }

  if (metaState.opacity > 0.6) {
    improvements.push({
      type: 'add_type_hints',
      description: 'Code clarity improvements: adding TypeScript types and documentation.',
      code: '// Add TypeScript strict types to all exported functions',
      riskLevel: 'low',
    })
  }

  if (metaState.distributionShift > 0.3) {
    improvements.push({
      type: 'adapt_ui',
      description:
        'User workflow has changed significantly. Optimizing UI layout for new usage patterns.',
      code: '// Reorganize UI based on new usage patterns',
      riskLevel: 'medium',
    })
  }

  if (metaState.anomalyScore > 0.5) {
    improvements.push({
      type: 'investigate_anomaly',
      description: 'Unusual usage patterns detected. Investigating potential issues.',
      code: '// Flag anomalous patterns for review',
      riskLevel: 'low',
    })
  }

  return improvements
}

/**
 * Apply an improvement to an app's code.
 */
export async function applyImprovement(
  appId: string,
  improvement: Improvement
): Promise<void> {
  if (improvement.riskLevel === 'low') {
    await prisma.improvement.create({
      data: {
        appId,
        type: improvement.type,
        description: improvement.description,
        code: improvement.code,
        status: 'applied',
        riskLevel: improvement.riskLevel,
        appliedAt: new Date(),
      },
    })
  } else {
    await prisma.improvement.create({
      data: {
        appId,
        type: improvement.type,
        description: improvement.description,
        code: improvement.code,
        status: 'suggested',
        riskLevel: improvement.riskLevel,
      },
    })
  }
}

// --- Helper functions ---

function shannonEntropy(values: number[]): number {
  const total = values.reduce((a, b) => a + b, 0)
  if (total === 0) return 0
  const probabilities = values.map((v) => v / total)
  return -probabilities.reduce((sum, p) => (p > 0 ? sum + p * Math.log(p) : sum), 0)
}

function getActionDistribution(analytics: Analytics[]): Record<string, number> {
  const dist: Record<string, number> = {}
  for (const a of analytics) {
    dist[a.actionType] = (dist[a.actionType] || 0) + 1
  }
  return dist
}

function klDivergence(p: Record<string, number>, q: Record<string, number>): number {
  const allKeys = new Set([...Object.keys(p), ...Object.keys(q)])
  const pTotal = Object.values(p).reduce((a, b) => a + b, 1)
  const qTotal = Object.values(q).reduce((a, b) => a + b, 1)

  let divergence = 0
  for (const key of allKeys) {
    const pProb = (p[key] || 0) / pTotal
    const qProb = (q[key] || 0) / qTotal

    if (pProb > 0) {
      divergence += pProb * Math.log(pProb / (qProb + 1e-10))
    }
  }

  return Math.min(1.0, divergence)
}
