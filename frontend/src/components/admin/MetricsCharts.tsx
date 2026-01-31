/**
 * Metrics Charts Component
 * ========================
 * 
 * Professional chart components for displaying metrics data.
 */

import { InformationCircleIcon } from '@heroicons/react/24/outline';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';

interface ChartDataPoint {
  timestamp?: string;
  time?: string;
  [key: string]: string | number | undefined;
}

interface SparklineProps {
  data: ChartDataPoint[];
  dataKey: string;
  color?: string;
  height?: number;
}

/**
 * Sparkline chart for inline metrics
 */
export function Sparkline({ data, dataKey, color = '#0284c7', height = 40 }: SparklineProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
        <defs>
          <linearGradient id={`gradient-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          fill={`url(#gradient-${dataKey})`}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

interface TimeSeriesChartProps {
  data: ChartDataPoint[];
  dataKeys: Array<{ key: string; name: string; color: string }>;
  title?: string;
  height?: number;
}

/**
 * Time series line chart
 */
export function TimeSeriesChart({ data, dataKeys, title, height = 300 }: TimeSeriesChartProps) {
  return (
    <div className="space-y-2">
      {title && <h3 className="font-semibold text-gray-900">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" stroke="#9ca3af" style={{ fontSize: '12px' }} />
          <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            labelStyle={{ color: '#111827' }}
          />
          <Legend />
          {dataKeys.map((dk) => (
            <Line
              key={dk.key}
              type="monotone"
              dataKey={dk.key}
              stroke={dk.color}
              dot={false}
              strokeWidth={2}
              name={dk.name}
              isAnimationActive={true}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface DistributionChartProps {
  data: ChartDataPoint[];
  dataKey: string;
  title?: string;
  height?: number;
}

/**
 * Distribution bar chart for latency histograms, etc.
 */
export function DistributionChart({ data, dataKey, title, height = 250 }: DistributionChartProps) {
  return (
    <div className="space-y-2">
      {title && <h3 className="font-semibold text-gray-900">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="bucket" stroke="#9ca3af" style={{ fontSize: '12px' }} />
          <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            labelStyle={{ color: '#111827' }}
          />
          <Bar dataKey={dataKey} fill="#0284c7" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface ComposedChartProps {
  data: ChartDataPoint[];
  bars: Array<{ key: string; name: string; color: string }>;
  lines: Array<{ key: string; name: string; color: string }>;
  title?: string;
  height?: number;
}

/**
 * Combined bar and line chart
 */
export function CombinedMetricsChart({ data, bars, lines, title, height = 300 }: ComposedChartProps) {
  return (
    <div className="space-y-2">
      {title && <h3 className="font-semibold text-gray-900">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" stroke="#9ca3af" style={{ fontSize: '12px' }} />
          <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} yAxisId="left" />
          <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} yAxisId="right" orientation="right" />
          <Tooltip
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            labelStyle={{ color: '#111827' }}
          />
          <Legend />
          {bars.map((bar) => (
            <Bar key={bar.key} dataKey={bar.key} fill={bar.color} name={bar.name} yAxisId="left" />
          ))}
          {lines.map((line) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              stroke={line.color}
              name={line.name}
              yAxisId="right"
              strokeWidth={2}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Information section for Metrics Charts
 */
export const MetricsChartsInfo = {
  title: 'System Metrics & Visualization',
  description: 'Monitor system performance with real-time charts and metrics',
  tips: [
    'Sparklines show quick trends for individual metrics at a glance',
    'Time series charts help identify patterns and anomalies over time',
    'Distribution charts reveal performance bucket distributions',
    'Combined charts compare two metrics on different scales',
    'Hover over charts to see exact values and timestamps'
  ],
  icon: InformationCircleIcon,
};
