import Anthropic from '@anthropic-ai/sdk'

const client = new Anthropic()

export interface GenerationRequest {
  appName: string
  appDescription: string
  features: string[]
  dataModel: Record<string, string[]>
}

export interface GeneratedCode {
  frontendCode: string
  backendCode: string
  databaseSchema: string
}

export async function generateAppCode(
  request: GenerationRequest
): Promise<GeneratedCode> {
  const [databaseSchema, backendCode, frontendCode] = await Promise.all([
    generateDatabaseSchema(request),
    generateBackendCode(request),
    generateFrontendCode(request),
  ])

  return { frontendCode, backendCode, databaseSchema }
}

async function generateDatabaseSchema(req: GenerationRequest): Promise<string> {
  const dataModelDesc = Object.entries(req.dataModel)
    .map(([entity, fields]) => `  ${entity}: ${fields.join(', ')}`)
    .join('\n')

  const prompt = `You are a database schema expert. Generate a PostgreSQL schema for this app:

App: ${req.appName}
Description: ${req.appDescription}
Features: ${req.features.join(', ')}

Data Model:
${dataModelDesc}

Generate:
1. CREATE TABLE statements with appropriate types
2. Indexes for performance
3. Foreign keys for relationships
4. Constraints for data integrity

Return ONLY the SQL code, no explanation.`

  const message = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 2000,
    messages: [{ role: 'user', content: prompt }],
  })

  return extractText(message)
}

async function generateBackendCode(req: GenerationRequest): Promise<string> {
  const prompt = `You are a Node.js/Express backend expert. Generate Express routes for this app:

App: ${req.appName}
Description: ${req.appDescription}
Features: ${req.features.join(', ')}

Generate:
1. Express route handlers (POST, GET, PUT, DELETE)
2. Request validation
3. Error handling
4. Database queries using Prisma ORM

Follow these patterns:
- Use async/await
- TypeScript types
- Proper HTTP status codes
- Standardized error responses

Return ONLY the TypeScript code, no explanation.`

  const message = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 3000,
    messages: [{ role: 'user', content: prompt }],
  })

  return extractText(message)
}

async function generateFrontendCode(req: GenerationRequest): Promise<string> {
  const prompt = `You are a React expert. Generate React components for this app:

App: ${req.appName}
Description: ${req.appDescription}
Features: ${req.features.join(', ')}

Generate:
1. React functional components using TypeScript
2. State management with useState and useEffect
3. API calls using fetch
4. Styling with Tailwind CSS classes
5. Form handling
6. Analytics tracking calls (trackAction function)

Include:
- Error boundaries
- Loading states
- Empty states
- Responsive design

Return ONLY the TypeScript/TSX code, no explanation.`

  const message = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4000,
    messages: [{ role: 'user', content: prompt }],
  })

  return extractText(message)
}

function extractText(message: Anthropic.Message): string {
  return message.content
    .filter((block): block is Anthropic.TextBlock => block.type === 'text')
    .map((block) => block.text)
    .join('')
}
