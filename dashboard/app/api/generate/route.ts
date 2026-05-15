import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ModalResult = {
  model?: string;
  seconds?: number;
  images?: { seed: number; data_url: string }[];
  controls_used?: number;
  error?: string;
};

function clampNumber(value: unknown, fallback: number, min: number, max: number) {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(Math.max(parsed, min), max);
}

function imageDimension(value: unknown) {
  const clamped = clampNumber(value, 768, 256, 1536);
  return Math.round(clamped / 16) * 16;
}

function cleanBody(value: unknown) {
  const body = value as Record<string, unknown>;
  const controlImages = Array.isArray(body.control_images)
    ? body.control_images
        .map((item) => {
          if (typeof item === "string") return item;
          if (item && typeof item === "object" && "data_url" in item) {
            return String((item as { data_url?: unknown }).data_url ?? "");
          }
          return "";
        })
        .filter((item) => item.startsWith("data:image/") && item.length < 12_000_000)
        .slice(0, 7)
    : [];

  return {
    prompt: String(body.prompt ?? "").trim(),
    count: Math.min(Math.max(Number(body.count ?? 1), 1), 2),
    seed: body.seed ? Number(body.seed) : undefined,
    width: imageDimension(body.width),
    height: imageDimension(body.height),
    steps: Math.round(clampNumber(body.steps, 50, 1, 75)),
    guidance_scale: clampNumber(body.guidance_scale, 4, 0, 10),
    control_images: controlImages,
  };
}

async function callModal(url: string | undefined, payload: ReturnType<typeof cleanBody>): Promise<ModalResult> {
  if (!url) return { error: "Missing Modal endpoint URL" };
  const endpoint = url.endsWith("/") ? url : `${url}/`;

  const headers: Record<string, string> = { "content-type": "application/json" };
  if (process.env.MODAL_INFERENCE_TOKEN) {
    headers.authorization = `Bearer ${process.env.MODAL_INFERENCE_TOKEN}`;
  }

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    const text = await response.text();
    let data: Record<string, unknown> = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { error: text };
    }

    if (!response.ok) {
      const message = data.detail ?? data.error ?? `Modal returned ${response.status}`;
      return { error: String(message).slice(0, 800) };
    }

    return data;
  } catch (error) {
    return { error: error instanceof Error ? error.message : "Modal request failed" };
  }
}

export async function POST(request: Request) {
  const payload = cleanBody(await request.json());
  if (!payload.prompt) {
    return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
  }

  const [base, finetuned] = await Promise.all([
    callModal(process.env.MODAL_BASE_FLUX_URL, payload),
    callModal(process.env.MODAL_FINETUNED_FLUX_URL, payload),
  ]);

  return NextResponse.json({ base, finetuned });
}
