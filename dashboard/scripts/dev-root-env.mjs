// Runs `next dev` with the repository's ../.env loaded, so local development reads the same
// DATABASE_URL as the Python tools without copying it into a second file.
import { spawn } from "node:child_process";

process.loadEnvFile("../.env");
const next = spawn(
  process.execPath,
  ["./node_modules/next/dist/bin/next", "dev", ...process.argv.slice(2)],
  { stdio: "inherit", env: process.env },
);
next.on("exit", (code) => process.exit(code ?? 0));
