/** @type {import('next').NextConfig} */
const nextConfig = {
  // Build to static files (the `out/` folder) so FastAPI can serve the UI from
  // the same origin as the API in the single-container deploy.
  output: "export",
  images: { unoptimized: true },
};
module.exports = nextConfig;
