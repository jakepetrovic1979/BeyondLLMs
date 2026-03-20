import { Router, Request, Response } from 'express'
import { prisma } from '../db'
import { generateAppCode } from '../services/codeGenerationService'

const router = Router()

router.post('/generate', async (req: Request, res: Response) => {
  try {
    const { appName, appDescription, features, dataModel } = req.body

    if (!appName || !appDescription) {
      return res.status(400).json({ error: 'appName and appDescription are required' })
    }

    console.log(`Generating code for: ${appName}`)

    const generatedCode = await generateAppCode({
      appName,
      appDescription,
      features: features || [],
      dataModel: dataModel || {},
    })

    const app = await prisma.app.create({
      data: {
        name: appName,
        description: appDescription,
        frontendCode: generatedCode.frontendCode,
        backendCode: generatedCode.backendCode,
        databaseSchema: generatedCode.databaseSchema,
        deploymentStatus: 'draft',
      },
    })

    // Initialize meta state
    await prisma.metaState.create({
      data: {
        appId: app.id,
        predictionError: 0.5,
        phi: 0.5,
        taskDifficulty: 0.3,
      },
    })

    res.json({ success: true, appId: app.id, app })
  } catch (error: any) {
    console.error('Generation error:', error)
    res.status(500).json({ error: error.message })
  }
})

export default router
