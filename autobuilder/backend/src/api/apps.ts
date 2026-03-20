import { Router, Request, Response } from 'express'
import { prisma } from '../db'

const router = Router()

router.post('/apps', async (req: Request, res: Response) => {
  try {
    const { userId, name, description, frontendCode, backendCode, databaseSchema } = req.body

    if (!userId || !name) {
      return res.status(400).json({ error: 'userId and name are required' })
    }

    const app = await prisma.app.create({
      data: {
        userId,
        name,
        description: description || '',
        frontendCode: frontendCode || '',
        backendCode: backendCode || '',
        databaseSchema: databaseSchema || '',
      },
    })

    // Initialize meta state
    await prisma.metaState.create({
      data: {
        appId: app.id,
        predictionError: 0.5,
        complexity: 0.5,
        entropy: 0.5,
        novelty: 0.5,
        phi: 0.5,
        opacity: 0.5,
        bindingStrength: 0.5,
        taskDifficulty: 0.3,
        distributionShift: 0.1,
        anomalyScore: 0.0,
      },
    })

    res.json(app)
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

router.get('/apps/:appId', async (req: Request, res: Response) => {
  try {
    const app = await prisma.app.findUnique({
      where: { id: req.params.appId as string },
      include: {
        metaState: true,
        analytics: { take: 100, orderBy: { timestamp: 'desc' } },
        improvements: { where: { status: 'suggested' } },
      },
    })

    if (!app) {
      return res.status(404).json({ error: 'App not found' })
    }

    res.json(app)
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

router.get('/apps', async (req: Request, res: Response) => {
  try {
    const userId = req.query.userId as string
    const where = userId ? { userId } : {}

    const apps = await prisma.app.findMany({
      where,
      include: { metaState: true },
      orderBy: { updatedAt: 'desc' },
    })

    res.json(apps)
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

router.post('/apps/:appId/analytics', async (req: Request, res: Response) => {
  try {
    const { type, data, error } = req.body

    await prisma.analytics.create({
      data: {
        appId: req.params.appId as string,
        actionType: type,
        actionData: data,
        errorMessage: error,
      },
    })

    res.json({ success: true })
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

export default router
