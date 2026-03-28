import { Agent } from "@mariozechner/pi-agent-core";
import { getModel } from "@mariozechner/pi-ai";

const DEFAULT_BASE_URL = "http://127.0.0.1:8400/v1";
const DEFAULT_MODEL = "gpt-4o-mini";

function parseArgs(argv) {
  let baseUrl = DEFAULT_BASE_URL;
  let runId = null;
  let modelId = DEFAULT_MODEL;

  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--base-url" && i + 1 < argv.length) {
      baseUrl = argv[++i];
    } else if (argv[i] === "--run-id" && i + 1 < argv.length) {
      runId = argv[++i];
    } else if (argv[i] === "--model" && i + 1 < argv.length) {
      modelId = argv[++i];
    }
  }

  return { baseUrl, runId, modelId };
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  const { baseUrl, runId, modelId } = parseArgs(process.argv);

  const raw = await readStdin();
  const task = JSON.parse(raw);

  // Use the model from task, CLI arg, or default
  const effectiveModel = task.model || modelId;

  // Get a known OpenAI model as template, then override baseUrl and model ID.
  // The proxy forwards whatever model name we send to the upstream.
  const model = getModel("openai", "gpt-4o-mini");
  if (!model) {
    throw new Error("Failed to get base model from pi-ai — is @mariozechner/pi-ai installed?");
  }
  model.baseUrl = baseUrl;
  model.id = effectiveModel; // Override: this is the model name sent to upstream
  model.api = "openai-completions"; // Force Chat Completions API (proxy endpoint)
  if (runId) {
    model.headers = { ...model.headers, "X-ClawLoop-Run-Id": runId };
  }

  process.stderr.write(`[runner] model=${effectiveModel} baseUrl=${baseUrl} runId=${runId}\n`);

  const agent = new Agent({
    initialState: { systemPrompt: task.instruction || "", model },
  });

  let output = "";
  agent.subscribe((event) => {
    process.stderr.write(`[runner] event: ${event.type}\n`);
    if (event.type === "message_end") {
      const msg = event.message;
      process.stderr.write(`[runner] message_end role=${msg?.role} content_type=${typeof msg?.content}\n`);
      if (msg?.content) {
        process.stderr.write(`[runner] content: ${JSON.stringify(msg.content).slice(0, 300)}\n`);
      }
      if (msg?.role === "assistant") {
        const content = msg.content;
        if (typeof content === "string") {
          output += content;
        } else if (Array.isArray(content)) {
          for (const block of content) {
            if (block.type === "text") {
              output += block.text;
            }
          }
        }
      }
    }
  });

  try {
    await agent.prompt(task.instruction || "Hello");
    await agent.waitForIdle();

    process.stdout.write(
      JSON.stringify({ task_id: task.task_id, status: "success", output }) + "\n"
    );
  } catch (err) {
    process.stderr.write(`[runner] error: ${err.message}\n`);
    process.stdout.write(
      JSON.stringify({
        task_id: task.task_id,
        status: "error",
        output: err.message,
      }) + "\n"
    );
    process.exit(1);
  }
}

main();
