/**
 * Admin: Benchmark Dashboard
 *
 * Displays COSE benchmark scores over time. Each time `run_all.py --save-to-api`
 * is invoked the new run appears as a data point on the charts.
 *
 * Route: /admin/benchmarks
 */

import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useBenchmarkRuns } from '../../hooks/useAdminAPI';
import { BenchmarkRun, BenchmarkResult } from '../../types/admin';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BENCH_CONFIGS: { key: string; label: string; metricKey: string; color: string }[] = [
  { key: 'conflict',    label: 'Conflict F1',      metricKey: 'f1',           color: '#6366f1' },
  { key: 'goal',        label: 'Goal Accuracy',     metricKey: 'accuracy',     color: '#10b981' },
  { key: 'recall',      label: 'Recall@10',         metricKey: 'recall_at_10', color: '#f59e0b' },
  { key: 'credibility', label: 'Credibility',       metricKey: 'auc',          color: '#ec4899' },
  { key: 'temporal',    label: 'Temporal',          metricKey: 'accuracy',     color: '#8b5cf6' },
  { key: 'latency',     label: 'P95 Latency (s)',   metricKey: 'p95_seconds',  color: '#ef4444' },
];

const STRATEGY_COLORS: Record<string, string> = {
  heuristic: '#0284c7',
  llm: '#16a34a',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getMetric(row: BenchmarkResult, key: string): number | undefined {
  const raw = row[key] ?? row[`${key}_mean`];
  return typeof raw === 'number' ? raw : undefined;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function buildTimeSeries(runs: BenchmarkRun[]) {
  // newest → oldest from API; reverse to oldest → newest for chart
  return [...runs].reverse().map((run) => {
    const point: Record<string, string | number | undefined> = {
      time: formatDate(run.run_at),
      composite: run.composite_score,
      strategy: run.strategy,
      model: run.vllm_model ?? 'heuristic',
    };
    for (const cfg of BENCH_CONFIGS) {
      const row = run.results.find((r) => r.benchmark === cfg.key);
      if (row) {
        point[cfg.key] = getMetric(row, cfg.metricKey);
      }
    }
    return point;
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface StatCardProps {
  label: string;
  value: string | number | undefined;
  unit?: string;
  color?: string;
}

function StatCard({ label, value, unit, color = '#6366f1' }: StatCardProps) {
  const displayVal = value !== undefined && value !== null ? String(value) : '—';
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex flex-col gap-1">
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</span>
      <span className="text-2xl font-bold" style={{ color }}>{displayVal}</span>
      {unit && <span className="text-xs text-gray-400">{unit}</span>}
    </div>
  );
}

interface MiniLineChartProps {
  data: ReturnType<typeof buildTimeSeries>;
  dataKey: string;
  label: string;
  color: string;
  domain?: [number | string, number | string];
}

function MiniLineChart({ data, dataKey, label, color, domain = [0, 1] }: MiniLineChartProps) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h4 className="text-sm font-semibold text-gray-700 mb-2">{label}</h4>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="time" tick={{ fontSize: 10 }} />
          <YAxis domain={domain} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(v: number | undefined) => [typeof v === 'number' ? v.toFixed(4) : v, label]}
          />
          {dataKey === 'composite' && (
            <ReferenceLine y={0.5} stroke="#d1d5db" strokeDasharray="4 4" />
          )}
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty / loading states
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-gray-400">
      <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <p className="text-lg font-medium">No benchmark runs yet</p>
      <p className="text-sm mt-1 text-center max-w-xs">
        Run the benchmark suite with <code className="bg-gray-100 px-1 rounded text-gray-600">--save-to-api</code> to populate this dashboard.
      </p>
      <pre className="mt-4 text-xs bg-gray-50 border border-gray-200 rounded p-3 text-gray-500 max-w-lg">
{`python -m tests.benchmarks.run_all \\
  --strategy heuristic \\
  --dataset kaggle \\
  --save-to-api http://localhost:8000/api/v1 \\
  --api-token <your-admin-token>`}
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

const STRATEGY_OPTIONS = ['all', 'heuristic', 'llm'] as const;

export default function BenchmarkDashboardPage() {
  const [strategyFilter, setStrategyFilter] = useState<string>('all');
  const { data: runs, isLoading, isError } = useBenchmarkRuns(
    100,
    strategyFilter === 'all' ? undefined : strategyFilter,
  );

  const timeSeries = useMemo(() => (runs ? buildTimeSeries(runs) : []), [runs]);
  const latest = runs?.[0];
  const latestResults = latest?.results ?? [];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-6 text-red-600 text-sm">
        Failed to load benchmark data. Check that the API is running.
      </div>
    );
  }

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Benchmark Dashboard</h1>
          <p className="text-sm text-gray-500 mt-1">
            COSE — Cognitive Operating System Evaluation · scores over time
          </p>
        </div>
        <select
          value={strategyFilter}
          onChange={(e) => setStrategyFilter(e.target.value)}
          className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {STRATEGY_OPTIONS.map((s) => (
            <option key={s} value={s}>{s === 'all' ? 'All strategies' : s}</option>
          ))}
        </select>
      </div>

      {(!runs || runs.length === 0) ? (
        <EmptyState />
      ) : (
        <>
          {/* Latest run summary cards */}
          <section>
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Latest run
              {latest && (
                <span className="ml-2 font-normal normal-case text-gray-400">
                  {new Date(latest.run_at).toLocaleString()} ·{' '}
                  {latest.strategy}{latest.vllm_model ? ` / ${latest.vllm_model}` : ''} ·{' '}
                  {latest.dataset} dataset
                </span>
              )}
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              <StatCard
                label="Composite"
                value={latest?.composite_score.toFixed(4)}
                color="#0284c7"
              />
              {BENCH_CONFIGS.map((cfg) => {
                const row = latestResults.find((r) => r.benchmark === cfg.key);
                const val = row ? getMetric(row, cfg.metricKey) : undefined;
                return (
                  <StatCard
                    key={cfg.key}
                    label={cfg.label}
                    value={val !== undefined ? val.toFixed(4) : undefined}
                    color={cfg.color}
                  />
                );
              })}
            </div>
          </section>

          {/* Composite score over time */}
          <section>
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Composite score over time
            </h2>
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={timeSeries} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ fontSize: 12 }}
                    formatter={(v: number | undefined, _: string | undefined, props: any) => [
                      (v ?? 0).toFixed(4),
                      `${props.payload.strategy}${props.payload.model !== 'heuristic' ? ' / ' + props.payload.model : ''}`,
                    ]}
                  />
                  <Legend />
                  <ReferenceLine y={0.5} stroke="#d1d5db" strokeDasharray="4 4" label={{ value: '0.5', fontSize: 10 }} />
                  <Line
                    type="monotone"
                    dataKey="composite"
                    name="Composite score"
                    stroke="#0284c7"
                    strokeWidth={2.5}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6 }}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Per-benchmark mini charts */}
          <section>
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Per-benchmark over time
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {BENCH_CONFIGS.map((cfg) => (
                <MiniLineChart
                  key={cfg.key}
                  data={timeSeries}
                  dataKey={cfg.key}
                  label={cfg.label}
                  color={cfg.color}
                  domain={cfg.key === 'latency' ? [0, 'auto'] : [0, 1]}
                />
              ))}
            </div>
          </section>

          {/* Run history table */}
          <section>
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Run history ({runs.length})
            </h2>
            <div className="overflow-x-auto rounded-xl border border-gray-100 shadow-sm">
              <table className="min-w-full divide-y divide-gray-100 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {['Date', 'Strategy', 'Model', 'Dataset', 'Duration', 'Composite', 'Conflict F1', 'Goal Acc', 'Recall@10'].map((h) => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-50">
                  {runs.map((run) => {
                    const r = (bench: string, key: string) => {
                      const row = run.results.find((x) => x.benchmark === bench);
                      const v = row ? getMetric(row, key) : undefined;
                      return v !== undefined ? v.toFixed(4) : '—';
                    };
                    return (
                      <tr key={run.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-2 whitespace-nowrap text-gray-700">
                          {new Date(run.run_at).toLocaleString()}
                        </td>
                        <td className="px-4 py-2">
                          <span
                            className="inline-block px-2 py-0.5 rounded-full text-xs font-medium"
                            style={{
                              background: STRATEGY_COLORS[run.strategy] + '20',
                              color: STRATEGY_COLORS[run.strategy],
                            }}
                          >
                            {run.strategy}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-gray-500">{run.vllm_model ?? '—'}</td>
                        <td className="px-4 py-2 text-gray-500">{run.dataset}</td>
                        <td className="px-4 py-2 text-gray-500">{run.duration_seconds.toFixed(1)}s</td>
                        <td className="px-4 py-2 font-semibold text-blue-600">{run.composite_score.toFixed(4)}</td>
                        <td className="px-4 py-2 text-gray-600">{r('conflict', 'f1')}</td>
                        <td className="px-4 py-2 text-gray-600">{r('goal', 'accuracy')}</td>
                        <td className="px-4 py-2 text-gray-600">{r('recall', 'recall_at_10')}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
