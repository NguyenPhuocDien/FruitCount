import { NextResponse } from 'next/server'
import { listLocalModels } from '@/lib/fruitcount-models'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function GET() {
  try { return NextResponse.json({ models: await listLocalModels() }) }
  catch (error) { return NextResponse.json({ error: (error as Error).message }, { status: 500 }) }
}
