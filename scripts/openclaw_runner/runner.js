import { Agent } from "@mariozechner/pi-agent-core";
import { getModel } from "@mariozechner/pi-ai";

const DEFAULT_BASE_URL = "http://127.0.0.1:8400/v1";

function parseArgs(argv) {
  let baseUrl = DEFAULT_BASE_URL;
  let runId = null;

  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--base-url" && i + 1 < argv.length) {
      baseUrl = argv[++i];
    } else if (argv[i] === "--run-id" && i + 1 < argv.length) {
      runId = argv[++i];
    }
  }

  return { baseUrl, runId };
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  const { baseUrl, runId } = parseArgs(process.argv);

  const raw = await readStdin();
  const task = JSON.parse(raw);

  const model = getModel("openai", "gpt-4o");
  model.baseUrl = baseUrl;
  if (runId) {
    model.headers = { "X-ClawLoop-Run-Id": runId };
  }

  const agent = new Agent({ initialState: { systemPrompt: task.instruction, model } });

  let output = "";
  agent.subscribe((event) => {
    if (event.type === "message_end" && event.message?.role === "assistant") {
      for (const block of event.message.content ?? []) {
        if (block.type === "text") {
          output += block.text;
        }
      }
    }
  });

  try {
    await agent.prompt(task.instruction);
    await agent.waitForIdle();

    process.stdout.write(JSON.stringify({ task_id: task.task_id, status: "success", output }) + "\n");
  } catch (err) {
    process.stdout.write(JSON.stringify({ task_id: task.task_id, status: "error", output: err.message }) + "\n");
    process.exit(1);
  }
}

main();
