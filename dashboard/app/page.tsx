"use client";

/* eslint-disable @next/next/no-img-element */

import { ChangeEvent, FormEvent, useState } from "react";

type ImageResult = { seed: number; data_url: string };
type ControlImage = { id: string; name: string; data_url: string };
type ModelResult = {
  model?: string;
  seconds?: number;
  images?: ImageResult[];
  controls_used?: number;
  error?: string;
};

type GenerateResponse = {
  base?: ModelResult;
  finetuned?: ModelResult;
  error?: string;
};

const EXAMPLE_PROMPT =
  "A manga panel of Character 1 standing in the foreground, sharp ink linework, city rooftops behind them, low camera angle, late afternoon light";

function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function FocusIcon() {
  return (
    <svg
      aria-hidden="true"
      className="mt-0.5 size-4 shrink-0"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
    </svg>
  );
}

function ResultColumn({ title, result, loading }: { title: string; result?: ModelResult; loading: boolean }) {
  return (
    <section className="min-w-0">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium text-zinc-950">{title}</h2>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          {result?.controls_used ? <span>{result.controls_used} ctrl</span> : null}
          {result?.seconds ? <span>{result.seconds}s</span> : null}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2">
        {loading ? (
          <div className="grid aspect-square place-items-center rounded-lg border border-zinc-200 bg-white text-sm text-zinc-500">
            Generating
          </div>
        ) : null}

        {!loading && result?.error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">{result.error}</div>
        ) : null}

        {!loading && result?.images?.length
          ? result.images.map((image, index) => (
              <figure
                key={`${image.seed}-${index}`}
                className="group relative justify-self-center overflow-hidden rounded-lg border border-zinc-200 bg-white"
              >
                <img
                  src={image.data_url}
                  alt={`${title} variant ${index + 1}`}
                  className="block h-auto max-h-[520px] w-auto max-w-full"
                />
                <figcaption className="absolute right-3 top-3 rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs font-medium text-zinc-950 opacity-0 shadow-sm transition-opacity group-hover:opacity-100">
                  seed {image.seed}
                </figcaption>
              </figure>
            ))
          : null}

        {!loading && !result ? (
          <div className="grid aspect-square place-items-center rounded-lg border border-dashed border-zinc-200 bg-white text-sm text-zinc-400">
            Waiting
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default function Home() {
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPT);
  const [count, setCount] = useState(1);
  const [seed, setSeed] = useState("");
  const [width, setWidth] = useState(768);
  const [height, setHeight] = useState(768);
  const [steps, setSteps] = useState(50);
  const [guidance, setGuidance] = useState(4);
  const [controls, setControls] = useState<ControlImage[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  async function addControls(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).filter((file) => file.type.startsWith("image/"));
    if (!files.length) return;
    const remaining = Math.max(0, 7 - controls.length);
    const selected = files.slice(0, remaining);
    const next = await Promise.all(
      selected.map(async (file) => ({
        id: `${file.name}-${file.lastModified}-${file.size}`,
        name: file.name,
        data_url: await fileToDataUrl(file),
      })),
    );
    setControls((current) => [...current, ...next].slice(0, 7));
    event.target.value = "";
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setResult(null);

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        prompt,
        count,
        seed: seed ? Number(seed) : undefined,
        width,
        height,
        steps,
        guidance_scale: guidance,
        control_images: controls.map((control) => control.data_url),
      }),
    });
    const data = await response.json();
    setResult(response.ok ? data : { error: data?.error ?? "Generation failed" });
    setLoading(false);
  }

  return (
    <main className="min-h-screen bg-zinc-50 px-4 py-10 text-zinc-950 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <form onSubmit={submit} className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm">
          <div className="grid grid-cols-[auto_1fr] items-start gap-3">
            <FocusIcon />
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={3}
              className="min-h-24 w-full resize-y border-0 bg-transparent text-sm leading-6 text-zinc-950 outline-none placeholder:text-zinc-400"
              placeholder="Prompt"
            />
          </div>

          <div className="mt-4 border-t border-zinc-100 pt-4">
            <div className="flex flex-wrap items-center gap-3">
              <label className="inline-flex h-9 cursor-pointer items-center rounded-full border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-950 shadow-sm transition hover:bg-zinc-50">
                Add controls
                <input type="file" accept="image/*" multiple className="hidden" onChange={addControls} />
              </label>
              {controls.length ? (
                <button
                  type="button"
                  onClick={() => setControls([])}
                  className="h-9 rounded-full border border-zinc-200 bg-white px-4 text-sm text-zinc-600 transition hover:text-zinc-950"
                >
                  Clear
                </button>
              ) : null}
              <span className="text-sm text-zinc-500">ctrl_img_1 first, refs after</span>
            </div>

            {controls.length ? (
              <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
                {controls.map((control, index) => (
                  <figure
                    key={control.id}
                    className="group relative h-24 w-24 shrink-0 overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50"
                  >
                    <img src={control.data_url} alt={control.name} className="h-full w-full object-cover" />
                    <figcaption className="absolute inset-x-1 bottom-1 rounded bg-white/90 px-1.5 py-0.5 text-center text-[11px] font-medium text-zinc-950 shadow-sm">
                      {index === 0 ? "ctrl_img_1" : `ref ${index}`}
                    </figcaption>
                    <button
                      type="button"
                      onClick={() => setControls((current) => current.filter((item) => item.id !== control.id))}
                      className="absolute right-1 top-1 grid h-6 w-6 place-items-center rounded-full bg-white text-sm leading-none text-zinc-950 opacity-0 shadow-sm transition group-hover:opacity-100"
                      aria-label={`Remove ${control.name}`}
                    >
                      x
                    </button>
                  </figure>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-zinc-100 pt-4">
            <div className="flex overflow-hidden rounded-full border border-zinc-200 bg-zinc-50 p-0.5">
              {[1, 2].map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setCount(value)}
                  className={`h-8 rounded-full px-4 text-sm font-medium transition ${
                    count === value ? "bg-zinc-950 text-white" : "text-zinc-600 hover:text-zinc-950"
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>

            <input
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
              inputMode="numeric"
              className="h-9 w-36 rounded-full border border-zinc-200 bg-white px-4 text-sm outline-none focus:border-zinc-400"
              placeholder="seed"
            />

            <label className="flex h-9 items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 text-sm text-zinc-500">
              W
              <input
                value={width}
                onChange={(event) => setWidth(Number(event.target.value))}
                type="number"
                min={256}
                max={1536}
                step={16}
                className="w-16 border-0 bg-transparent text-zinc-950 outline-none"
              />
            </label>

            <label className="flex h-9 items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 text-sm text-zinc-500">
              H
              <input
                value={height}
                onChange={(event) => setHeight(Number(event.target.value))}
                type="number"
                min={256}
                max={1536}
                step={16}
                className="w-16 border-0 bg-transparent text-zinc-950 outline-none"
              />
            </label>

            <label className="flex h-9 items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 text-sm text-zinc-500">
              steps
              <input
                value={steps}
                onChange={(event) => setSteps(Number(event.target.value))}
                type="number"
                min={1}
                max={75}
                className="w-12 border-0 bg-transparent text-zinc-950 outline-none"
              />
            </label>

            <label className="flex h-9 items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 text-sm text-zinc-500">
              cfg
              <input
                value={guidance}
                onChange={(event) => setGuidance(Number(event.target.value))}
                type="number"
                min={0}
                max={10}
                step={0.5}
                className="w-12 border-0 bg-transparent text-zinc-950 outline-none"
              />
            </label>

            <button
              type="submit"
              disabled={loading || !prompt.trim()}
              className="ml-auto h-9 rounded-full bg-zinc-950 px-5 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Running" : "Generate"}
            </button>
          </div>
        </form>

        {result?.error ? <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">{result.error}</div> : null}

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <ResultColumn title="Base FLUX" result={result?.base} loading={loading} />
          <ResultColumn title="Fine-tuned checkpoint" result={result?.finetuned} loading={loading} />
        </div>
      </div>
    </main>
  );
}
