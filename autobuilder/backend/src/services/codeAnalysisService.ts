export interface CodeIssue {
  type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  location: string
  suggestion: string
}

export interface CodeIssues {
  missingTypeHints: CodeIssue[]
  complexFunctions: CodeIssue[]
  performanceIssues: CodeIssue[]
  missingErrorHandling: CodeIssue[]
  securityIssues: CodeIssue[]
}

export function analyzeCode(code: string): CodeIssues {
  return {
    missingTypeHints: findMissingTypeHints(code),
    complexFunctions: findComplexFunctions(code),
    performanceIssues: findPerformanceIssues(code),
    missingErrorHandling: findMissingErrorHandling(code),
    securityIssues: findSecurityIssues(code),
  }
}

export function getAllIssues(issues: CodeIssues): CodeIssue[] {
  return [
    ...issues.missingTypeHints,
    ...issues.complexFunctions,
    ...issues.performanceIssues,
    ...issues.missingErrorHandling,
    ...issues.securityIssues,
  ]
}

function findMissingTypeHints(code: string): CodeIssue[] {
  const issues: CodeIssue[] = []

  // Find function declarations without return types
  const functionRegex = /function\s+(\w+)\s*\([^)]*\)\s*\{/g
  let match

  while ((match = functionRegex.exec(code)) !== null) {
    const beforeFunc = code.substring(Math.max(0, match.index - 30), match.index)
    if (!beforeFunc.includes(':') || !beforeFunc.includes(')')) {
      issues.push({
        type: 'missing_type_hints',
        severity: 'low',
        location: `function ${match[1]}`,
        suggestion: `Add return type annotation to function ${match[1]}`,
      })
    }
  }

  return issues
}

function findComplexFunctions(code: string): CodeIssue[] {
  const issues: CodeIssue[] = []

  // Detect deeply nested code (3+ levels)
  const lines = code.split('\n')
  let maxIndent = 0
  let currentFunction = ''

  for (const line of lines) {
    const funcMatch = line.match(/function\s+(\w+)/)
    if (funcMatch) currentFunction = funcMatch[1]

    const indent = line.search(/\S/)
    if (indent > maxIndent) maxIndent = indent

    if (indent >= 12 && currentFunction) {
      issues.push({
        type: 'deep_nesting',
        severity: 'medium',
        location: `function ${currentFunction}`,
        suggestion: `Extract nested logic in ${currentFunction} into helper functions`,
      })
      break
    }
  }

  return issues
}

function findPerformanceIssues(code: string): CodeIssue[] {
  const issues: CodeIssue[] = []

  // N+1 query pattern
  if (code.includes('.find(') && (code.includes('for ') || code.includes('.forEach('))) {
    issues.push({
      type: 'n_plus_1_query',
      severity: 'high',
      location: 'Database queries in loop',
      suggestion: 'Use bulk queries or join instead of loop + find',
    })
  }

  // Missing pagination
  if (code.includes('.findMany(') && !code.includes('take:') && !code.includes('limit')) {
    issues.push({
      type: 'missing_pagination',
      severity: 'medium',
      location: 'Database query without limit',
      suggestion: 'Add pagination (take/skip) to findMany queries',
    })
  }

  return issues
}

function findMissingErrorHandling(code: string): CodeIssue[] {
  const issues: CodeIssue[] = []

  // Async functions without try-catch
  const asyncFuncRegex = /async\s+function\s+(\w+)/g
  let match

  while ((match = asyncFuncRegex.exec(code)) !== null) {
    const funcBody = extractFunctionBody(code, match.index)
    if (funcBody && !funcBody.includes('try') && !funcBody.includes('catch')) {
      issues.push({
        type: 'missing_error_handling',
        severity: 'high',
        location: `async function ${match[1]}`,
        suggestion: `Add try-catch block to async function ${match[1]}`,
      })
    }
  }

  return issues
}

function findSecurityIssues(code: string): CodeIssue[] {
  const issues: CodeIssue[] = []

  // String concatenation in queries
  if (code.includes('query(') && code.includes("'+")) {
    issues.push({
      type: 'sql_injection',
      severity: 'critical',
      location: 'String concatenation in query',
      suggestion: 'Use parameterized queries instead of string concatenation',
    })
  }

  // innerHTML usage
  if (code.includes('innerHTML') || code.includes('dangerouslySetInnerHTML')) {
    issues.push({
      type: 'xss_risk',
      severity: 'high',
      location: 'Direct HTML insertion',
      suggestion: 'Sanitize HTML content before insertion to prevent XSS',
    })
  }

  // Hardcoded secrets
  const secretPatterns = [/password\s*=\s*['"][^'"]+['"]/i, /api_key\s*=\s*['"][^'"]+['"]/i]
  for (const pattern of secretPatterns) {
    if (pattern.test(code)) {
      issues.push({
        type: 'hardcoded_secret',
        severity: 'critical',
        location: 'Hardcoded credential',
        suggestion: 'Move secrets to environment variables',
      })
    }
  }

  return issues
}

function extractFunctionBody(code: string, startIndex: number): string | null {
  let braceCount = 0
  let started = false
  let bodyStart = startIndex

  for (let i = startIndex; i < code.length; i++) {
    if (code[i] === '{') {
      if (!started) bodyStart = i
      braceCount++
      started = true
    } else if (code[i] === '}') {
      braceCount--
      if (started && braceCount === 0) {
        return code.substring(bodyStart, i + 1)
      }
    }
  }

  return null
}
