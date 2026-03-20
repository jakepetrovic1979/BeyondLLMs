import { Router, Request, Response } from 'express'
import { prisma } from '../db'

const router = Router()

router.get('/health', async (_req: Request, res: Response) => {
  let dbHealthy = false
  try {
    await prisma.$queryRaw`SELECT 1`
    dbHealthy = true
  } catch {
    // DB not ready yet — still return 200 so Railway knows the process is alive
  }

  res.json({
    status: dbHealthy ? 'healthy' : 'degraded',
    database: dbHealthy ? 'connected' : 'disconnected',
    timestamp: new Date(),
  })
})

export default router
