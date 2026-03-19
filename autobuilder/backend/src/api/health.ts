import { Router, Request, Response } from 'express'
import { prisma } from '../db'

const router = Router()

router.get('/health', async (_req: Request, res: Response) => {
  try {
    await prisma.$queryRaw`SELECT 1`

    const appCount = await prisma.app.count()
    const analyticsCount = await prisma.analytics.count()

    res.json({
      status: 'healthy',
      timestamp: new Date(),
      metrics: {
        apps: appCount,
        totalAnalytics: analyticsCount,
      },
    })
  } catch (error: any) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
    })
  }
})

export default router
