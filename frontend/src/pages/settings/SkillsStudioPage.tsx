import { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useCanApproveSkills, useCanEditSkills } from '@/stores/auth';

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
  active_version: string;
  submitted?: {
    submitted_at?: string;
    submitted_by_user_id?: string;
    status?: string;
  } | null;
  versions: Array<{
    version: string;
    approved_at?: string;
    source?: string;
  }>;
}

interface CoreAgentRow {
  agent_name: string;
  is_core: boolean;
}

interface SkillsStudioResponse {
  total_agents: number;
  core_agents: CoreAgentRow[];
  non_core_agents: AgentSkillRow[];
  can_edit: boolean;
  can_approve: boolean;
  last_published_at?: string | null;
  last_published_by_user_id?: string | null;
}

export function SkillsStudioPage() {
  const canEditSkills = useCanEditSkills();
  const canApproveSkills = useCanApproveSkills();
  const [data, setData] = useState<SkillsStudioResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPublishing, setIsPublishing] = useState(false);
  const [savingAgent, setSavingAgent] = useState<string | null>(null);
  const [submittingAgent, setSubmittingAgent] = useState<string | null>(null);
  const [approvingAgent, setApprovingAgent] = useState<string | null>(null);
  const [rollingBackAgent, setRollingBackAgent] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, AgentSkillDraft>>({});
  const [rollbackTargets, setRollbackTargets] = useState<Record<string, string>>({});

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

      const targets: Record<string, string> = {};
      for (const row of res.data.non_core_agents) {
        targets[row.agent_name] = 'v1';
      }
      setRollbackTargets(targets);
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
    if (!canApproveSkills) return;
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

  async function submitAgent(agentName: string) {
    if (!canEditSkills) return;
    setSubmittingAgent(agentName);
    try {
      await apiClient.post(`/admin/skills-studio/${agentName}/submit`);
      toast.success(`Submitted ${agentName} for admin approval`);
      await load();
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setSubmittingAgent(null);
    }
  }

  async function approveAgent(agentName: string) {
    if (!canApproveSkills) return;
    setApprovingAgent(agentName);
    try {
      const res = await apiClient.post(`/admin/skills-studio/${agentName}/approve`);
      toast.success(`Approved ${agentName} as ${res.data.version}`);
      await load();
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setApprovingAgent(null);
    }
  }

  async function rollbackAgent(agentName: string) {
    if (!canApproveSkills) return;
    const targetVersion = rollbackTargets[agentName] || 'v1';
    setRollingBackAgent(agentName);
    try {
      await apiClient.post(`/admin/skills-studio/${agentName}/rollback`, {
        target_version: targetVersion,
      });
      toast.success(`Rolled back ${agentName} using ${targetVersion}`);
      await load();
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setRollingBackAgent(null);
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
          disabled={isPublishing || !canApproveSkills}
          title={canApproveSkills ? 'Publish all drafts as new versions' : 'Only admins can publish'}
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
                <p className="text-xs text-gray-500 mt-1">Active version: {row.active_version}</p>
                <p className="text-xs text-gray-500 mt-1 whitespace-pre-wrap">
                  {row.published_skill.instructions || 'No published instructions yet.'}
                </p>
                {row.submitted && (
                  <p className="text-xs text-amber-700 mt-2">
                    Submission pending since {row.submitted.submitted_at ? new Date(row.submitted.submitted_at).toLocaleString() : 'now'}
                  </p>
                )}
                <p className="text-xs text-gray-500 mt-2">
                  Versions: {row.versions.map((v) => v.version).join(', ')}
                </p>
              </div>

              <div className="flex flex-wrap items-center justify-end gap-2">
                <button
                  className="btn-secondary"
                  onClick={() => saveAgent(row.agent_name)}
                  disabled={savingAgent === row.agent_name || !canEditSkills}
                >
                  {savingAgent === row.agent_name ? 'Saving…' : 'Save Draft'}
                </button>

                <button
                  className="btn-secondary"
                  onClick={() => submitAgent(row.agent_name)}
                  disabled={submittingAgent === row.agent_name || !canEditSkills}
                  title={canEditSkills ? 'Submit this draft to admin for approval' : 'Developer/admin role required'}
                >
                  {submittingAgent === row.agent_name ? 'Submitting…' : 'Submit for Approval'}
                </button>

                {canApproveSkills && (
                  <>
                    <button
                      className="btn-primary"
                      onClick={() => approveAgent(row.agent_name)}
                      disabled={approvingAgent === row.agent_name || !row.submitted}
                      title={row.submitted ? 'Approve submitted draft as next version' : 'No submitted draft'}
                    >
                      {approvingAgent === row.agent_name ? 'Approving…' : 'Approve'}
                    </button>

                    <select
                      className="input w-28"
                      value={rollbackTargets[row.agent_name] || 'v1'}
                      onChange={(e) => {
                        const value = e.target.value;
                        setRollbackTargets((prev) => ({ ...prev, [row.agent_name]: value }));
                      }}
                    >
                      {row.versions.map((v) => (
                        <option key={v.version} value={v.version}>{v.version}</option>
                      ))}
                    </select>

                    <button
                      className="btn-danger"
                      onClick={() => rollbackAgent(row.agent_name)}
                      disabled={rollingBackAgent === row.agent_name}
                    >
                      {rollingBackAgent === row.agent_name ? 'Rolling back…' : 'Rollback'}
                    </button>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
