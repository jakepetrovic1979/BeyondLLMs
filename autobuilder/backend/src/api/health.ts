import { Router, Request, Response } from 'express'
import { prisma } from '../db'

const router = Router()

router.get('/health', async (_req: Request, res: Response) => {
  try {
    await prisma.$queryRaw`SELECT 1`

    res.json({
      status: 'healthy',
      timestamp: new Date(),
    })
  } catch (error: any) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
    })
  }
})

export default router
