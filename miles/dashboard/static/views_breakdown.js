import { api } from "./api.js";
import { el, fmtNum, setViewCleanup } from "./app.js";
import { hideTooltip, SERIES_COLORS, showTooltip } from "./charts.js";

// A family is a set of logged metrics whose keys differ in one path segment,
// which is how a run splits a quantity by a label (a data source, a task, an
// outcome): `train_batch/source_share/<source>`. The Metrics view draws one
// line chart per key; this view puts the members of one family side by side.
//
// flat:    `<prefix>/<member>`          one bar per member
// stacked: `<prefix>/<member>/<part>`   one bar per member, one segment per part
//
// Families are found from the key names alone, so a prefix whose children are
// unrelated quantities is listed too. That is harmless: the reader picks the
// family. A prefix needs two segments, so that a whole category is not one.
export function metricFamilies(metricKeys) {
  const flat = new Map(); // prefix -> Map(member -> key)
  const stacked = new Map(); // prefix -> Map(member -> Map(part -> key))
  for (const key of metricKeys) {
    const segments = key.split("/");
    if (segments.length >= 3) {
      const prefix = segments.slice(0, -1).join("/");
      if (!flat.has(prefix)) flat.set(prefix, new Map());
      flat.get(prefix).set(segments.at(-1), key);
    }
    if (segments.length >= 4) {
      const prefix = segments.slice(0, -2).join("/");
      if (!stacked.has(prefix)) stacked.set(prefix, new Map());
      const members = stacked.get(prefix);
      const member = segments.at(-2);
      if (!members.has(member)) members.set(member, new Map());
      members.get(member).set(segments.at(-1), key);
    }
  }
  const families = [];
  for (const [prefix, members] of stacked) {
    const parts = new Set();
    for (const partKeys of members.values()) for (const part of partKeys.keys()) parts.add(part);
    if (members.size < 2 || parts.size < 2) continue;
    const keys = [...members.values()].flatMap((partKeys) => [...partKeys.values()]);
    families.push({
      id: `${prefix}/*/*`,
      prefix,
      stacked: true,
      members: [...members.keys()].sort(),
      parts: [...parts].sort(),
      keys,
      keyOf: (member, part) => members.get(member)?.get(part),
    });
  }
  for (const [prefix, members] of flat) {
    if (members.size < 2) continue;
    families.push({
      id: `${prefix}/*`,
      prefix,
      stacked: false,
      members: [...members.keys()].sort(),
      parts: [null],
      keys: [...members.values()],
      keyOf: (member) => members.get(member),
    });
  }
  return families.sort((a, b) => a.id.localeCompare(b.id));
}

const AGGREGATES = {
  total: { label: "total over steps", of: (ys) => ys.reduce((sum, y) => sum + y, 0) },
  mean: { label: "mean of logged steps", of: (ys) => ys.reduce((sum, y) => sum + y, 0) / ys.length },
  last: { label: "last step", of: (ys) => ys.at(-1) },
};

const percent = (value, whole) => (whole > 0 ? `${((100 * value) / whole).toFixed(1)}%` : "—");

export async function renderBreakdown(view, meta, route) {
  const families = metricFamilies(meta.metric_keys);
  if (!families.length) {
    view.replaceChildren(el("p", { class: "muted" }, ["no metric is logged per label (keys like a/b/<label>)"]));
    return;
  }
  const family =
    families.find((f) => f.id === route.family) ??
    families.find((f) => f.id === sessionStorage.getItem("breakdownFamily")) ??
    families[0];
  const aggregate = AGGREGATES[route.aggregate] ? route.aggregate : "total";
  const normalize = family.stacked && route.normalize;
  const stepFrom = route.stepFrom === null ? null : Number(route.stepFrom);
  const stepTo = route.stepTo === null ? null : Number(route.stepTo);

  // every control rewrites the URL, so a view can be linked as it is shown
  const go = (changes) => {
    const state = {
      family: family.id,
      agg: aggregate,
      norm: normalize ? "1" : null,
      from: route.stepFrom,
      to: route.stepTo,
      ...changes,
    };
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(state)) if (v !== null && v !== "") params.set(k, v);
    location.replace(`#/breakdown?${params}`);
  };
  const select = (options, current, onchange) =>
    el(
      "select",
      { onchange: (ev) => onchange(ev.target.value) },
      options.map(([value, label]) => el("option", { value, ...(value === current ? { selected: "" } : {}) }, [label])),
    );
  const stepInput = (value, placeholder, name) =>
    el("input", { type: "number", min: "0", placeholder, value, onchange: (ev) => go({ [name]: ev.target.value }) });

  const xKey = meta.step_key_of_metric_key?.[family.keys[0]] ?? "rollout/step";
  const controls = el("div", { class: "controls" }, [
    select(
      families.map((f) => [f.id, `${f.id}  (${f.stacked ? `${f.members.length} × ${f.parts.length}` : f.members.length})`]),
      family.id,
      (id) => {
        sessionStorage.setItem("breakdownFamily", id);
        go({ family: id, norm: null });
      },
    ),
    select(
      Object.entries(AGGREGATES).map(([value, { label }]) => [value, label]),
      aggregate,
      (value) => go({ agg: value }),
    ),
    el("span", { class: "muted" }, [`${xKey} from`]),
    stepInput(route.stepFrom, "first", "from"),
    el("span", { class: "muted" }, ["to"]),
    stepInput(route.stepTo, "last", "to"),
    ...(family.stacked
      ? [
          el("label", { class: "muted", style: "display: flex; align-items: center; gap: 4px; cursor: pointer" }, [
            el("input", {
              type: "checkbox",
              ...(normalize ? { checked: "" } : {}),
              onchange: (ev) => go({ norm: ev.target.checked ? "1" : null }),
            }),
            "each bar to 100%",
          ]),
        ]
      : []),
  ]);
  const legend = el("div", { class: "legend", style: "flex-wrap: wrap; margin-bottom: 8px" });
  const status = el("p", { class: "muted" }, ["loading…"]);
  const bars = el("div", { class: "bars" });
  view.replaceChildren(el("div", { class: "panel" }, [el("h3", {}, ["Breakdown"]), controls, legend, status, bars]));

  if (family.stacked) {
    legend.replaceChildren(
      ...family.parts.map((part, i) =>
        el("span", { style: "display: flex; align-items: center; gap: 4px" }, [
          el("span", { class: "bar", style: `width: 10px; background: ${SERIES_COLORS[i % SERIES_COLORS.length]}` }),
          part,
        ]),
      ),
    );
  }

  let signature = null;
  async function load() {
    const series = await api("/api/metrics", { keys: family.keys.join(","), x: xKey });
    const steps = new Set();
    const valueOf = (key) => {
      const s = series[key];
      if (!s) return null;
      const ys = [];
      s.x.forEach((x, i) => {
        if (s.y[i] === null || (stepFrom !== null && x < stepFrom) || (stepTo !== null && x > stepTo)) return;
        steps.add(x);
        ys.push(s.y[i]);
      });
      return ys.length ? AGGREGATES[aggregate].of(ys) : null;
    };
    const rows = family.members
      .map((member) => {
        const values = family.parts.map((part) => valueOf(family.keyOf(member, part)));
        return { member, values, sum: values.reduce((sum, v) => sum + Math.max(v ?? 0, 0), 0) };
      })
      .filter((row) => row.values.some((v) => v !== null))
      .sort((a, b) => b.sum - a.sum);
    const next = JSON.stringify(rows);
    if (next === signature) return; // a live page repaints only when the numbers moved
    signature = next;

    const whole = rows.reduce((sum, row) => sum + row.sum, 0);
    const longest = Math.max(...rows.map((row) => row.sum), 0);
    status.textContent = rows.length
      ? `${rows.length} of ${family.members.length} members, ${steps.size} steps` +
        (aggregate === "total" ? `; total ${fmtNum(whole)}, shares are of that total` : "")
      : "nothing logged in this step range";
    bars.replaceChildren(
      ...rows.map((row) => {
        const scale = normalize ? row.sum : longest;
        const segments = row.values.map((value, i) => {
          if (value === null || value <= 0 || !(scale > 0)) return null;
          const part = family.parts[i];
          const text = `${part === null ? row.member : `${row.member} · ${part}`}\n${AGGREGATES[aggregate].label} = ${fmtNum(value)}${
            family.stacked ? `\n${percent(value, row.sum)} of this bar` : ""
          }`;
          return el("div", {
            class: "bar-seg",
            style: `width: ${(100 * value) / scale}%; background: ${SERIES_COLORS[i % SERIES_COLORS.length]}`,
            onmousemove: (ev) => showTooltip(ev.clientX, ev.clientY, text),
            onmouseleave: hideTooltip,
          });
        });
        const single = row.values.length === 1 ? row.values[0] : row.sum;
        const figure =
          aggregate === "total" && !normalize ? `${fmtNum(single)}  ${percent(row.sum, whole)}` : fmtNum(single);
        return el("div", { class: "bar-row" }, [
          el("div", { class: "bar-label", title: row.member }, [row.member]),
          el("div", { class: "bar-track" }, segments.filter(Boolean)),
          el("div", { class: "bar-value" }, [figure]),
        ]);
      }),
    );
  }

  const guarded = () =>
    load().catch((err) => {
      if (signature === null) status.textContent = String(err);
    });
  await guarded();
  if (meta.mode === "follow") {
    const intervalId = setInterval(guarded, 5000);
    setViewCleanup(() => clearInterval(intervalId));
  }
}
