#!/bin/sh
# Initialize Prisma on the bind-mounted SQLite file, then start the image UI.
# An empty aitk_db.db has no schema until this runs (ostris/ai-toolkit#820).
set -eu
cd /app/ai-toolkit/ui
./node_modules/.bin/prisma db push --skip-generate
exec /opt/toolkit-mcp/bin/python -m aitoolkit_mcp.supervise
