import React, { useEffect, useMemo, useState } from 'react';
import { apiClient } from '@/lib/api';

type Summary = Record<string, number>;
type Point = { date: string; value: number };

const METRICS = ['memory_writes', 'cognitive_gateway_calls'];

const UsageDashboard: React.FC = () => {
  const [days, setDays] = useState(30);
  const [summary, setSummary] = useState<Summary>({});
  const [series, setSeries] = useState<Record<string, Point[]>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const [summaryRes, ...dailyRes] = await Promise.all([
          apiClient.get('/admin/usage/summary', { params: { days } }),
          ...METRICS.map(metric =>
            apiClient.get('/admin/usage/daily', { params: { metric, days } })
          ),
        ]);

        if (cancelled) {
          return;
        }

        setSummary(summaryRes.data.summary || {});
        const nextSeries: Record<string, Point[]> = {};
        dailyRes.forEach((res, idx) => {
          nextSeries[METRICS[idx]] = res.data.points || [];
        });
        setSeries(nextSeries);
      } catch {
        if (!cancelled) {
          setError('Failed to load usage data');
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [days]);

  const cards = useMemo(
    () =>
      METRICS.map(metric => ({
        metric,
        total: summary[metric] || 0,
      })),
    [summary]
  );

  if (isLoading) {
    return <div className="p-6">Loading usage dashboard...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-700">{error}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Usage Dashboard</h1>
          <p className="text-gray-600 mt-1">Organization usage over time</p>
        </div>
        <select
          value={days}
          onChange={e => setDays(Number(e.target.value))}
          className="border border-gray-300 rounded-md px-3 py-2"
        >
          <option value={7}>7 days</option>
          <option value={30}>30 days</option>
          <option value={90}>90 days</option>
        </select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {cards.map(card => (
          <div key={card.metric} className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-sm text-gray-600">{card.metric}</div>
            <div className="text-3xl font-semibold text-gray-900">{card.total}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {METRICS.map(metric => (
          <div key={metric} className="bg-white border border-gray-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">{metric}</h2>
            <div className="space-y-2 max-h-72 overflow-auto">
              {(series[metric] || []).map(point => (
                <div key={`${metric}-${point.date}`} className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">{point.date}</span>
                  <span className="font-medium text-gray-900">{point.value}</span>
                </div>
              ))}
              {(series[metric] || []).length === 0 && (
                <div className="text-sm text-gray-500">No data in selected window.</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UsageDashboard;
