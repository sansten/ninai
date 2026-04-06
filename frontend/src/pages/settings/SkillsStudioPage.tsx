import { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { apiClient, getErrorMessage } from '@/lib/api';

interface AgentSkillDraft {
  enabled: boolean;
  instructions: string;
  parameters: Record<string, unknown>;
}

interface AgentSkillRow {
  agent_name: string;
  is_core: boolean;
  skill: AgentSkillDraft;
  published_skill: AgentSkillDraft;
}

interface CoreAgentRow {
  agent_name: string;
  is_core: boolean;
}

interface SkillsStudioResponse {
  total_agents: number;
  core_agents: CoreAgentRow[];
  non_core_agents: AgentSkillRow[];
  last_published_at?: string | null;
  last_published_by_user_id?: string | null;
}

export function SkillsStudioPage() {
  const [data, setData] = useState<SkillsStudioResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPublishing, setIsPublishing] = useState(false);
  const [savingAgent, setSavingAgent] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, AgentSkillDraft>>({});

  async function load() {
    setIsLoading(true);
    try {
      const res = await apiClient.get<SkillsStudioResponse>('/admin/skills-studio');
      setData(res.data);
      const nextDrafts: Record<string, AgentSkillDraft> = {};
      for (const row of res.data.non_core_agents) {
        nextDrafts[row.agent_name] = {
          enabled: row.skill.enabled,
          instructions: row.skill.instructions,
          parameters: row.skill.parameters ?? {},
        };
      }
      setDrafts(nextDrafts);
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const nonCoreAgents = useMemo(() => data?.non_core_agents ?? [], [data]);

  async function saveAgent(agentName: string) {
    const draft = drafts[agentName];
    if (!draft) return;

    setSavingAgent(agentName);
    try {
      await apiClient.put(`/admin/skills-studio/${agentName}`, {
        enabled: draft.enabled,
        instructions: draft.instructions,
        parameters: draft.parameters,
      });
      toast.success(`Saved draft for ${agentName}`);
      await load();
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setSavingAgent(null);
    }
  }

  async function publishAll() {
    setIsPublishing(true);
    try {
      const res = await apiClient.post('/admin/skills-studio/publish');
      toast.success(`Published ${res.data.published_count} non-core agent skills`);
      await load();
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setIsPublishing(false);
    }
  }

  if (isLoading) {
    return <div className="text-gray-500">Loading skills studio…</div>;
  }

  if (!data) {
    return <div className="text-red-600">Unable to load skills studio data.</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Skills Studio</h1>
          <p className="text-sm text-gray-500 mt-1">
            Customize skill overlays for non-core agents and publish to your organization.
          </p>
          <p className="text-xs text-gray-400 mt-2">
            Total agents: {data.total_agents} • Core: {data.core_agents.length} • Non-core: {data.non_core_agents.length}
          </p>
          {data.last_published_at && (
            <p className="text-xs text-gray-400 mt-1">
              Last published: {new Date(data.last_published_at).toLocaleString()}
            </p>
          )}
        </div>
        <button
          className="btn-primary"
          onClick={publishAll}
          disabled={isPublishing}
        >
          {isPublishing ? 'Publishing…' : 'Publish Skills'}
        </button>
      </div>

      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900">Core Agents (Read-only)</h2>
        <p className="text-sm text-gray-500 mt-1">Core agents are protected and cannot be edited in Skills Studio.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {data.core_agents.map((agent) => (
            <span key={agent.agent_name} className="badge-gray">{agent.agent_name}</span>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        {nonCoreAgents.map((row) => {
          const draft = drafts[row.agent_name] ?? row.skill;
          return (
            <div key={row.agent_name} className="card space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-gray-900">{row.agent_name}</h3>
                <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={draft.enabled}
                    onChange={(e) => {
                      setDrafts((prev) => ({
                        ...prev,
                        [row.agent_name]: {
                          ...draft,
                          enabled: e.target.checked,
                        },
                      }));
                    }}
                  />
                  Enabled
                </label>
              </div>

              <div>
                <label className="label">Skill Instructions</label>
                <textarea
                  className="input min-h-[120px]"
                  value={draft.instructions}
                  onChange={(e) => {
                    setDrafts((prev) => ({
                      ...prev,
                      [row.agent_name]: {
                        ...draft,
                        instructions: e.target.value,
                      },
                    }));
                  }}
                  placeholder="Describe how this non-core agent should adapt behavior for your organization."
                />
              </div>

              <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                <p className="text-xs font-medium text-gray-700">Published Snapshot</p>
                <p className="text-xs text-gray-500 mt-1 whitespace-pre-wrap">
                  {row.published_skill.instructions || 'No published instructions yet.'}
                </p>
              </div>

              <div className="flex justify-end">
                <button
                  className="btn-secondary"
                  onClick={() => saveAgent(row.agent_name)}
                  disabled={savingAgent === row.agent_name}
                >
                  {savingAgent === row.agent_name ? 'Saving…' : 'Save Draft'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
