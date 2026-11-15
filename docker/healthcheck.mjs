const response = await fetch(`http://127.0.0.1:${process.env.PORT || 3000}/api/health`, {
  signal: AbortSignal.timeout(4000),
})
if (!response.ok) process.exit(1)
const data = await response.json()
if (data.status !== 'ok') process.exit(1)
