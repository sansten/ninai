/**
 * Memory Insights Page
 *
 * Professional analytics dashboard for the memory corpus. Every chart and
 * metric carries an info icon (ⓘ) that opens a plain-English tooltip
 * explaining exactly what is measured, why it matters, and what each
 * colour / legend item represents.
 */

import { useState, useRef, useEffect, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { InformationCircleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { apiClient } from '@/lib/api';
import { useAuthStore } from '@/stores/auth';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TrendPoint { date: string; count: number }
interface DistributionItem { label: string; count: number }
interface AgentCoverage { agent_name: string; memory_count: number }

interface InsightsSummary {
  org_id: string;
  total_memories: number;
  active_memories: number;
  inactive_memories: number;
  pending_review: number;
  enriched_memories: number;
  creation_trend: TrendPoint[];
  scope_distribution: DistributionItem[];
  type_distribution: DistributionItem[];
  agent_coverage: AgentCoverage[];
  retrieved_at: string;
}

// ---------------------------------------------------------------------------
// Agent catalogue — short name → plain-English role description
// ---------------------------------------------------------------------------

const AGENT_DESCRIPTIONS: Record<string, string> = {
  SemanticNormalization:  'Normalises phrasing and vocabulary so memories with the same meaning are linked correctly.',
  EntityResolution:       'Identifies people, organisations, and places mentioned and links them across memories.',
  ContextAmplifier:       'Adds surrounding context and background knowledge to make sparse memories more useful.',
  SiloPropagation:        'Shares relevant signal from one team or department silo to others that would benefit.',
  OrgAttention:           'Scores which memories deserve the most organisational attention right now.',
  ProactiveMemoryPush:    'Proactively surfaces memories to users before they think to search for them.',
  CausalReasoning:        'Builds cause-and-effect chains between events so you can understand why things happened.',
  ConflictDetection:      'Flags memories that contradict each other across different silos.',
  AdaptiveConflictResolution: 'Resolves flagged conflicts automatically, picking the most credible version.',
  MemoryDecay:            'Ages out stale or superseded memories to keep the corpus healthy and relevant.',
  MemoryConsolidation:    'Merges near-duplicate memories into a single canonical record.',
  TemporalReasoning:      'Understands timestamps, sequences, and time-gaps between related events.',
  EpisodicGrouping:       'Groups related memories into episodes — coherent stories about a period or event.',
  Credibility:            'Scores how trustworthy each memory is based on source, corroboration, and history.',
  Playbook:               'Matches memories to known procedures and runbooks your organisation has encoded.',
  GoalDecomposition:      'Breaks high-level goals into concrete sub-tasks and maps memories to each step.',
  UncertaintyReporting:   'Reports how confident the system is about a memory and flags ones needing review.',
  NarrativeSynthesis:     'Composes a human-readable narrative summary from a collection of raw memories.',
  FeedbackIntegration:    'Incorporates human review verdicts back into future enrichment passes.',
  AnomalyDetection:       'Spots unusual patterns — outliers, sudden spikes, or unexpected behaviour.',
  QueryIntelligence:      'Improves search queries by inferring intent, extracting entities, and adding filters.',
};

// ---------------------------------------------------------------------------
// Palette & scope/type metadata
// ---------------------------------------------------------------------------

const PALETTE = ['#0284c7', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0891b2', '#be185d', '#065f46'];

const SCOPE_DESCRIPTIONS: Record<string, string> = {
  personal:     'Only the owner can see this memory.',
  team:         'Shared with a specific team.',
  department:   'Shared across an entire department.',
  division:     'Visible to a business division.',
  organization: 'Visible to everyone in the organisation.',
  global:       'Shared across all organisations (platform-level).',
};

const TYPE_DESCRIPTIONS: Record<string, string> = {
  short_term:  'Temporary context — recent events or transient observations. Ages out quickly.',
  long_term:   'Durable knowledge — stable facts and institutional memory. Low decay rate.',
  semantic:    'Conceptual knowledge — what things mean, how they relate. Used for reasoning.',
  procedural:  'How-to knowledge — steps, runbooks, and playbooks. Used for task execution.',
};

// ---------------------------------------------------------------------------
// InfoTooltip — hover the ⓘ icon to read a plain-English explanation
// ---------------------------------------------------------------------------

interface InfoTooltipProps {
  content: ReactNode;
}

function InfoTooltip({ content }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div ref={ref} className="relative inline-flex items-center">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="ml-1.5 text-gray-400 hover:text-primary-500 focus:outline-none transition-colors"
        aria-label="More information"
      >
        <InformationCircleIcon className="h-4 w-4" />
      </button>

      {open && (
        <div className="absolute left-0 top-6 z-50 w-72 rounded-xl border border-gray-200 bg-white p-4 shadow-xl text-sm text-gray-700 leading-relaxed">
          {content}
          <div className="absolute -top-1.5 left-2 h-3 w-3 rotate-45 border-l border-t border-gray-200 bg-white" />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SectionCard — consistent panel wrapper with title + info icon
// ---------------------------------------------------------------------------

interface SectionCardProps {
  title: string;
  info: ReactNode;
  children: ReactNode;
  className?: string;
}

function SectionCard({ title, info, children, className = '' }: SectionCardProps) {
  return (
    <div className={`rounded-xl border border-gray-200 bg-white p-5 shadow-sm ${className}`}>
      <div className="flex items-center gap-1 mb-1">
        <h2 className="text-sm font-semibold text-gray-800">{title}</h2>
        <InfoTooltip content={info} />
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LegendDot — colour swatch + label + optional description
// ---------------------------------------------------------------------------

interface LegendDotProps {
  color: string;
  label: string;
  description?: string;
  count?: number;
  total?: number;
}

function LegendDot({ color, label, description, count, total }: LegendDotProps) {
  const pct = count !== undefined && total ? Math.round((count / total) * 100) : undefined;
  return (
    <div className="flex items-start gap-2 text-sm">
      <span
        className="mt-0.5 h-3 w-3 shrink-0 rounded-sm"
        style={{ backgroundColor: color }}
      />
      <span className="text-gray-700 font-medium min-w-[90px]">{label}</span>
      {description && <span className="text-gray-400 text-xs leading-relaxed">{description}</span>}
      {count !== undefined && (
        <span className="ml-auto text-gray-500 font-medium whitespace-nowrap">
          {count.toLocaleString()}{pct !== undefined ? ` (${pct}%)` : ''}
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Custom Recharts tooltip
// ---------------------------------------------------------------------------

interface ChartTooltipProps {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number | string; color?: string }>;
  label?: string;
  unit?: string;
}

function ChartTooltip({ active, payload, label, unit = '' }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-lg text-xs">
      {label && <p className="font-semibold text-gray-700 mb-1">{label}</p>}
      {payload.map((entry) => (
        <p key={entry.name} className="text-gray-600">
          <span className="font-medium" style={{ color: entry.color ?? '#0284c7' }}>
            {entry.name}:
          </span>{' '}
          {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
          {unit}
        </p>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------

interface KpiCardProps {
  label: string;
  value: string | number;
  sub?: string;
  info: ReactNode;
  accentColor?: string;
}

function KpiCard({ label, value, sub, info, accentColor = '#0284c7' }: KpiCardProps) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-1">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</p>
        <InfoTooltip content={info} />
      </div>
      <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
      {sub && (
        <p className="mt-1 text-xs text-gray-400">{sub}</p>
      )}
      <div className="mt-3 h-0.5 w-8 rounded-full" style={{ backgroundColor: accentColor }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shortName(name: string) {
  return name.replace(/Agent$/, '');
}

const DAYS_OPTIONS = [7, 14, 30, 60, 90] as const;
type DaysOption = typeof DAYS_OPTIONS[number];

async function fetchInsights(days: number): Promise<InsightsSummary> {
  const res = await apiClient.get<InsightsSummary>(`/memories/insights?days=${days}`);
  return res.data;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function MemoryInsightsPage() {
  const { currentOrg } = useAuthStore();
  const queryClient = useQueryClient();
  const [days, setDays] = useState<DaysOption>(30);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  const { data, isLoading, isError, isFetching } = useQuery<InsightsSummary>({
    queryKey: ['memory-insights', currentOrg?.id, days],
    queryFn: () => fetchInsights(days),
    staleTime: 60_000,
    enabled: !!currentOrg,
  });

  function handleRefresh() {
    queryClient.invalidateQueries({ queryKey: ['memory-insights', currentOrg?.id, days] });
    setLastRefreshed(new Date());
  }

  function handleDaysChange(d: DaysOption) {
    setDays(d);
    setLastRefreshed(new Date());
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
        Failed to load memory insights. Check that the backend is running and try refreshing.
      </div>
    );
  }

  const enrichmentPct = data.total_memories > 0
    ? Math.round((data.enriched_memories / data.total_memories) * 100)
    : 0;

  const trendData = data.creation_trend.map((p) => ({
    date: p.date.slice(5),   // "MM-DD"
    'New memories': p.count,
  }));

  const lifecycleSteps = [
    { label: 'Created (total)',       value: data.total_memories,    color: '#0284c7',
      desc: 'Every memory ever written to this organisation.' },
    { label: 'Enriched',             value: data.enriched_memories,  color: '#7c3aed',
      desc: 'Processed by at least one enrichment agent (entity resolution, causal reasoning, etc.).' },
    { label: 'Active',               value: data.active_memories,    color: '#059669',
      desc: 'Still considered current and retrievable by agents.' },
    { label: 'Inactive / decayed',   value: data.inactive_memories,  color: '#d97706',
      desc: 'Aged out by the Memory Decay agent — stale or superseded knowledge.' },
    { label: 'Pending human review', value: data.pending_review,     color: '#dc2626',
      desc: 'Flagged as uncertain or conflicted; waiting for a human to approve or reject.' },
  ];

  return (
    <div className="space-y-8">
      {/* ── Page header ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Memory Insights</h1>
          <p className="mt-1 text-sm text-gray-500">
            A live view of your organisation's memory corpus — how it grows, how agents enrich it, and where it needs attention.
            Hover any <InformationCircleIcon className="inline h-4 w-4 text-gray-400" /> icon for a plain-English explanation.
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Trend window selector */}
          <div className="flex items-center gap-1 rounded-lg border border-gray-200 bg-white p-1 shadow-sm">
            {DAYS_OPTIONS.map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => handleDaysChange(d)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  days === d
                    ? 'bg-primary-600 text-white shadow-sm'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {d}d
              </button>
            ))}
          </div>

          {/* Refresh button + last-refreshed */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleRefresh}
              disabled={isFetching}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 shadow-sm hover:bg-gray-50 disabled:opacity-50"
            >
              <ArrowPathIcon className={`h-3.5 w-3.5 ${isFetching ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <span className="text-xs text-gray-400 hidden sm:block">
              Updated {lastRefreshed.toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>

      {/* ── KPI row ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <KpiCard
          label="Total Memories"
          value={data.total_memories.toLocaleString()}
          sub="all time"
          accentColor="#0284c7"
          info={
            <div className="space-y-1">
              <p className="font-semibold text-gray-900">Total Memories</p>
              <p>The complete count of every memory record ever written to your organisation — active, inactive, and pending review combined.</p>
              <p className="text-gray-500">Use this as your baseline when comparing all other metrics.</p>
            </div>
          }
        />
        <KpiCard
          label="Active"
          value={data.active_memories.toLocaleString()}
          sub={`${data.inactive_memories.toLocaleString()} inactive`}
          accentColor="#059669"
          info={
            <div className="space-y-1">
              <p className="font-semibold text-gray-900">Active Memories</p>
              <p>Memories currently marked as <span className="font-medium">is_active = true</span> — they are live and retrievable by agents and search.</p>
              <p className="text-gray-500 mt-1"><span className="font-medium text-gray-600">Inactive</span> = aged out by the Memory Decay agent. They still exist in the database for audit purposes but are no longer surfaced in normal retrieval.</p>
            </div>
          }
        />
        <KpiCard
          label="Enriched"
          value={`${enrichmentPct}%`}
          sub={`${data.enriched_memories.toLocaleString()} memories`}
          accentColor="#7c3aed"
          info={
            <div className="space-y-1">
              <p className="font-semibold text-gray-900">Enrichment Coverage</p>
              <p>Percentage of memories that have been processed by at least one of the 21 enrichment agents (e.g. EntityResolution, CausalReasoning, CredibilityAgent).</p>
              <p className="text-gray-500 mt-1">A low percentage means newly written memories are still queued for enrichment — this is normal immediately after a bulk import.</p>
            </div>
          }
        />
        <KpiCard
          label="Pending Review"
          value={data.pending_review.toLocaleString()}
          sub="queued for human approval"
          accentColor="#dc2626"
          info={
            <div className="space-y-1">
              <p className="font-semibold text-gray-900">Pending Human Review</p>
              <p>Memories the <span className="font-medium">HumanReviewQueueAgent</span> has flagged because they are uncertain, conflicted, or require clearance to promote.</p>
              <p className="text-gray-500 mt-1">High numbers here indicate the system encountered a lot of ambiguous or conflicting data recently. Navigate to <span className="font-medium">Review Queue</span> to process them.</p>
            </div>
          }
        />
        <KpiCard
          label="Enrichment Agents"
          value={data.agent_coverage.length}
          sub="distinct agents active"
          accentColor="#d97706"
          info={
            <div className="space-y-1">
              <p className="font-semibold text-gray-900">Active Enrichment Agents</p>
              <p>How many of the 21 enrichment agents have logged at least one successful run against your corpus.</p>
              <p className="text-gray-500 mt-1">If this is lower than expected, some agents may be disabled or their upstream dependencies (e.g. Qdrant, vLLM) may be unavailable.</p>
            </div>
          }
        />
      </div>

      {/* ── Row 2: Creation trend + Scope distribution ───────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

        {/* Creation trend */}
        <SectionCard
          title={`Memory Creation — Last ${days} Days`}
          info={
            <div className="space-y-2">
              <p className="font-semibold text-gray-900">Memory Creation Trend</p>
              <p>Number of new memory records written each day over the last 30 days.</p>
              <div className="mt-2 space-y-1">
                <p className="font-medium text-gray-700">What to look for:</p>
                <ul className="list-disc pl-4 space-y-0.5 text-gray-500">
                  <li>A steady upward trend means your team is actively capturing knowledge.</li>
                  <li>A sudden spike could mean a bulk import or an automated pipeline firing.</li>
                  <li>A flat line means no new memories were written — worth investigating.</li>
                </ul>
              </div>
              <p className="text-gray-500 mt-2 text-xs">X-axis = date (MM-DD). Y-axis = count of new memories.</p>
            </div>
          }
        >
          {trendData.length === 0 ? (
            <p className="mt-4 text-sm text-gray-400">No memories written in this window yet.</p>
          ) : (
            <>
              <div className="mt-4">
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                    <defs>
                      <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#0284c7" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#0284c7" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="date" stroke="#9ca3af" style={{ fontSize: '10px' }} interval="preserveStartEnd" tick={{ dy: 4 }} />
                    <YAxis stroke="#9ca3af" style={{ fontSize: '10px' }} allowDecimals={false} />
                    <RechartsTooltip content={<ChartTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="New memories"
                      stroke="#0284c7"
                      strokeWidth={2}
                      fill="url(#trendGrad)"
                      dot={false}
                      activeDot={{ r: 4, fill: '#0284c7' }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              {/* Inline legend */}
              <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
                <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
                <span><span className="font-medium text-gray-700">New memories</span> — count of memory records created on that date</span>
              </div>
            </>
          )}
        </SectionCard>

        {/* Scope distribution */}
        <SectionCard
          title="Scope Distribution"
          info={
            <div className="space-y-2">
              <p className="font-semibold text-gray-900">Memory Scope</p>
              <p>Scope controls who can see a memory. The bar chart shows how many memories exist at each visibility level.</p>
              <div className="mt-2 space-y-1.5">
                {Object.entries(SCOPE_DESCRIPTIONS).map(([scope, desc]) => (
                  <div key={scope} className="text-xs">
                    <span className="font-medium text-gray-700 capitalize">{scope}: </span>
                    <span className="text-gray-500">{desc}</span>
                  </div>
                ))}
              </div>
              <p className="text-gray-500 mt-2 text-xs">Ideally most memories are <span className="font-medium">organization</span>-scoped — that maximises cross-team knowledge sharing.</p>
            </div>
          }
        >
          {data.scope_distribution.length === 0 ? (
            <p className="mt-4 text-sm text-gray-400">No data yet.</p>
          ) : (
            <>
              <div className="mt-4">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={data.scope_distribution}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 5, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={false} />
                    <XAxis type="number" stroke="#9ca3af" style={{ fontSize: '10px' }} allowDecimals={false} />
                    <YAxis type="category" dataKey="label" stroke="#9ca3af" style={{ fontSize: '10px' }} width={80} />
                    <RechartsTooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const val = payload[0].value as number;
                        const desc = SCOPE_DESCRIPTIONS[label as string] ?? '';
                        return (
                          <div className="rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-lg text-xs max-w-[200px]">
                            <p className="font-semibold capitalize text-gray-800">{label}</p>
                            <p className="text-gray-500 mt-0.5 leading-snug">{desc}</p>
                            <p className="mt-1.5 font-medium text-gray-700">{val.toLocaleString()} memories</p>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="count" radius={[0, 6, 6, 0]} name="Memories">
                      {data.scope_distribution.map((_, i) => (
                        <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              {/* Colour legend */}
              <div className="mt-3 space-y-1.5">
                {data.scope_distribution.map((item, i) => (
                  <LegendDot
                    key={item.label}
                    color={PALETTE[i % PALETTE.length]}
                    label={item.label}
                    description={SCOPE_DESCRIPTIONS[item.label]}
                    count={item.count}
                    total={data.total_memories}
                  />
                ))}
              </div>
            </>
          )}
        </SectionCard>
      </div>

      {/* ── Row 3: Memory type + Lifecycle funnel ───────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

        {/* Memory type distribution */}
        <SectionCard
          title="Memory Type Distribution"
          info={
            <div className="space-y-2">
              <p className="font-semibold text-gray-900">Memory Types</p>
              <p>Ninai classifies every memory into one of four cognitive types, each with a different decay rate and retrieval purpose.</p>
              <div className="mt-2 space-y-1.5">
                {Object.entries(TYPE_DESCRIPTIONS).map(([type, desc]) => (
                  <div key={type} className="text-xs">
                    <span className="font-medium text-gray-700 capitalize">{type.replace('_', ' ')}: </span>
                    <span className="text-gray-500">{desc}</span>
                  </div>
                ))}
              </div>
              <p className="text-gray-500 mt-2 text-xs">A healthy corpus has a mix of long-term and procedural memories as its backbone, with short-term memories cycling through regularly.</p>
            </div>
          }
        >
          {data.type_distribution.length === 0 ? (
            <p className="mt-4 text-sm text-gray-400">No data yet.</p>
          ) : (
            <div className="mt-4 space-y-4">
              {data.type_distribution.map((item, i) => {
                const pct = data.total_memories > 0
                  ? Math.round((item.count / data.total_memories) * 100)
                  : 0;
                const desc = TYPE_DESCRIPTIONS[item.label];
                return (
                  <div key={item.label}>
                    <div className="flex items-center justify-between text-sm mb-1.5 gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-sm shrink-0"
                          style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                        />
                        <span className="font-medium text-gray-700 capitalize">
                          {item.label.replace('_', ' ')}
                        </span>
                      </div>
                      <span className="text-gray-500 font-medium whitespace-nowrap">
                        {item.count.toLocaleString()} ({pct}%)
                      </span>
                    </div>
                    <div className="h-2.5 w-full rounded-full bg-gray-100 overflow-hidden">
                      <div
                        className="h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, backgroundColor: PALETTE[i % PALETTE.length] }}
                      />
                    </div>
                    {desc && (
                      <p className="mt-1 text-xs text-gray-400 leading-relaxed">{desc}</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </SectionCard>

        {/* Lifecycle funnel */}
        <SectionCard
          title="Memory Lifecycle"
          info={
            <div className="space-y-2">
              <p className="font-semibold text-gray-900">Memory Lifecycle Funnel</p>
              <p>Shows how memories flow through Ninai's pipeline from initial write to final state. Each bar is a subset of the one above it.</p>
              <div className="mt-2 space-y-1.5">
                {lifecycleSteps.map((step) => (
                  <div key={step.label} className="text-xs flex gap-2">
                    <span className="h-2.5 w-2.5 rounded-sm shrink-0 mt-0.5" style={{ backgroundColor: step.color }} />
                    <div>
                      <span className="font-medium text-gray-700">{step.label}: </span>
                      <span className="text-gray-500">{step.desc}</span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-gray-500 mt-2 text-xs">A large <span className="font-medium text-amber-600">inactive</span> bar relative to total is normal in a mature corpus — it means the decay agent is doing its job.</p>
            </div>
          }
        >
          <div className="mt-4 space-y-4">
            {lifecycleSteps.map((step) => {
              const pct = data.total_memories > 0
                ? Math.round((step.value / data.total_memories) * 100)
                : 0;
              return (
                <div key={step.label}>
                  <div className="flex items-center justify-between text-sm mb-1.5 gap-2">
                    <div className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-sm shrink-0" style={{ backgroundColor: step.color }} />
                      <span className="font-medium text-gray-700">{step.label}</span>
                    </div>
                    <span className="text-gray-500 font-medium whitespace-nowrap">
                      {step.value.toLocaleString()} ({pct}%)
                    </span>
                  </div>
                  <div className="h-2.5 w-full rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className="h-2.5 rounded-full transition-all duration-500"
                      style={{ width: `${pct}%`, backgroundColor: step.color }}
                    />
                  </div>
                  <p className="mt-1 text-xs text-gray-400 leading-relaxed">{step.desc}</p>
                </div>
              );
            })}
          </div>
        </SectionCard>
      </div>

      {/* ── Agent enrichment coverage ────────────────────────────────────── */}
      <SectionCard
        title="Enrichment Agent Coverage"
        info={
          <div className="space-y-2">
            <p className="font-semibold text-gray-900">Enrichment Agent Coverage</p>
            <p>Each bar shows how many <span className="font-medium">distinct memories</span> a given enrichment agent has successfully processed in this organisation.</p>
            <p className="text-gray-500">Hover a bar to see the agent's full name and role. Agents with low counts may be newly deployed or waiting on upstream data.</p>
            <div className="mt-2 border-t border-gray-100 pt-2">
              <p className="font-medium text-gray-700 mb-1">Agent name legend:</p>
              <p className="text-gray-500 text-xs">The chart strips the word "Agent" from names for readability. For example, "Credibility" = CredibilityAgent.</p>
            </div>
          </div>
        }
      >
        {data.agent_coverage.length === 0 ? (
          <p className="mt-4 text-sm text-gray-400">No agent runs recorded yet. Start the enrichment pipeline to see data here.</p>
        ) : (
          <>
            <div className="mt-4">
              <ResponsiveContainer
                width="100%"
                height={Math.max(300, data.agent_coverage.length * 38)}
              >
                <BarChart
                  data={data.agent_coverage.map((a) => ({
                    ...a,
                    display_name: shortName(a.agent_name),
                  }))}
                  layout="vertical"
                  margin={{ top: 5, right: 40, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={false} />
                  <XAxis type="number" stroke="#9ca3af" style={{ fontSize: '10px' }} allowDecimals={false} />
                  <YAxis
                    type="category"
                    dataKey="display_name"
                    stroke="#9ca3af"
                    style={{ fontSize: '10px' }}
                    width={175}
                    tick={{ fill: '#374151' }}
                  />
                  <RechartsTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const row = payload[0].payload as { display_name: string; memory_count: number };
                      const desc = AGENT_DESCRIPTIONS[row.display_name] ?? 'Enrichment agent.';
                      return (
                        <div className="rounded-lg border border-gray-200 bg-white px-3 py-2.5 shadow-lg text-xs max-w-[240px]">
                          <p className="font-semibold text-gray-800">{row.display_name}Agent</p>
                          <p className="text-gray-500 mt-1 leading-relaxed">{desc}</p>
                          <p className="mt-2 font-medium text-gray-700">
                            {row.memory_count.toLocaleString()} memories processed
                          </p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="memory_count" radius={[0, 6, 6, 0]} name="Memories processed">
                    {data.agent_coverage.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Agent legend — name + one-line role */}
            <div className="mt-4 border-t border-gray-100 pt-4">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Agent roles</p>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {data.agent_coverage.map((agent, i) => {
                  const short = shortName(agent.agent_name);
                  return (
                    <div key={agent.agent_name} className="flex items-start gap-2 text-xs">
                      <span
                        className="mt-0.5 h-2.5 w-2.5 shrink-0 rounded-sm"
                        style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                      />
                      <div>
                        <span className="font-medium text-gray-700">{short}: </span>
                        <span className="text-gray-400">
                          {AGENT_DESCRIPTIONS[short] ?? 'Enrichment agent.'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </SectionCard>

      {/* ── Footer timestamp ─────────────────────────────────────────────── */}
      <p className="text-right text-xs text-gray-400">
        Data as of {new Date(data.retrieved_at).toLocaleString()}
      </p>
    </div>
  );
}
