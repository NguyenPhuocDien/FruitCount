/** @type {import('next').NextConfig} */
const nextConfig = {
  // Preserve local launcher/Render behavior; Docker uses the minimal standalone server.
  output: process.env.NEXT_OUTPUT === 'standalone' ? 'standalone' : undefined,
  distDir: process.env.NEXT_DIST_DIR || '.next',
  // Model code/weights are mounted at runtime; they are not Node dependencies.
  outputFileTracingExcludes: {
    '/api/*': ['./backend FruitCount/**/*', './docs/**/*'],
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
