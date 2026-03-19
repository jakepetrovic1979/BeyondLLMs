import { prisma } from '../db'
import { computeMetaState, decideImprovements, applyImprovement } from './systemMService'

/**
 * Main System M loop that runs periodically.
 *
 * Flow:
 * 1. Collect analytics from all apps
 * 2. Compute meta-states
 * 3. Decide improvements
 * 4. Apply safe improvements / suggest risky ones
 * 5. Learn from outcomes
 */
export async function runSystemMLoop(): Promise<void> {
  console.log(`[System M] Loop started at ${new Date().toISOString()}`)

  const apps = await prisma.app.findMany()

  for (const app of apps) {
    try {
      // Step 1: Collect analytics
      const analytics = await prisma.analytics.findMany({
        where: { appId: app.id },
        orderBy: { timestamp: 'desc' },
        take: 1000,
      })

      if (analytics.length === 0) continue

      // Step 2: Compute meta-states
      const metaState = await computeMetaState(app.id, analytics)

      // Save meta-state
      await prisma.metaState.upsert({
        where: { appId: app.id },
        create: { appId: app.id, ...metaState },
        update: metaState,
      })

      // Step 3: Decide improvements
      const improvements = await decideImprovements(metaState)

      // Step 4: Apply safe improvements, suggest risky ones
      for (const improvement of improvements) {
        await applyImprovement(app.id, improvement)
      }

      console.log(
        `[System M] Processed app ${app.id}: ` +
        `PE=${metaState.predictionError.toFixed(2)} ` +
        `Phi=${metaState.phi.toFixed(2)} ` +
        `improvements=${improvements.length}`
      )
    } catch (error) {
      console.error(`[System M] Error processing app ${app.id}:`, error)
    }
  }

  console.log(`[System M] Loop completed at ${new Date().toISOString()}`)
}

/**
 * Evaluate whether a previously applied improvement actually helped.
 */
export async function evaluateImprovementOutcome(
  appId: string,
  improvementId: string
): Promise<{ success: boolean; delta: number } | null> {
  const improvement = await prisma.improvement.findUnique({
    where: { id: improvementId },
  })

  if (!improvement || improvement.status !== 'applied' || !improvement.appliedAt) {
    return null
  }

  const beforeAnalytics = await prisma.analytics.findMany({
    where: {
      appId,
      timestamp: { lt: improvement.appliedAt },
    },
    orderBy: { timestamp: 'desc' },
    take: 100,
  })

  const afterAnalytics = await prisma.analytics.findMany({
    where: {
      appId,
      timestamp: { gte: improvement.appliedAt },
    },
    take: 100,
  })

  if (beforeAnalytics.length === 0 || afterAnalytics.length === 0) {
    return null
  }

  const beforeErrorRate =
    beforeAnalytics.filter((a) => a.errorMessage).length / beforeAnalytics.length
  const afterErrorRate =
    afterAnalytics.filter((a) => a.errorMessage).length / afterAnalytics.length

  const delta = beforeErrorRate - afterErrorRate

  return {
    success: delta > 0.05,
    delta,
  }
}

/**
 * Start the System M loop on a schedule.
 */
export function startSystemMScheduler(intervalHours: number = 4): NodeJS.Timeout {
  console.log(`[System M] Scheduler started, running every ${intervalHours} hours`)

  // Run once immediately
  runSystemMLoop().catch(console.error)

  // Then schedule periodic runs
  return setInterval(
    () => runSystemMLoop().catch(console.error),
    intervalHours * 60 * 60 * 1000
  )
}
