import { Router, Request, Response } from 'express'
import { prisma } from '../db'

const router = Router()

router.get('/apps/:appId/improvements/suggested', async (req: Request, res: Response) => {
  try {
    const improvements = await prisma.improvement.findMany({
      where: {
        appId: req.params.appId as string,
        status: 'suggested',
      },
      orderBy: { createdAt: 'desc' },
    })

    res.json(improvements)
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

router.post('/improvements/:improvementId/approve', async (req: Request, res: Response) => {
  try {
    const improvement = await prisma.improvement.update({
      where: { id: req.params.improvementId as string },
      data: {
        status: 'applied',
        appliedAt: new Date(),
      },
    })

    // Apply the code change to the app
    const app = await prisma.app.findUnique({
      where: { id: improvement.appId },
    })

    if (app) {
      // For now, append improvement code as a comment
      // In production, use AST manipulation
      const updatedCode = `${app.frontendCode}\n\n// Applied improvement: ${improvement.type}\n${improvement.code}`

      await prisma.app.update({
        where: { id: app.id },
        data: { frontendCode: updatedCode },
      })
    }

    res.json({ success: true, improvement })
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

router.post('/improvements/:improvementId/reject', async (req: Request, res: Response) => {
  try {
    await prisma.improvement.update({
      where: { id: req.params.improvementId as string },
      data: { status: 'rejected' },
    })

    res.json({ success: true })
  } catch (error: any) {
    res.status(500).json({ error: error.message })
  }
})

export default router
