/**
 * Settings Management Page
 * Configure system settings by category
 */

import React, { useState } from 'react';
import { useAdmin } from '../../hooks/useAdmin';
import { useSettings, useCreateSetting, useUpdateSetting, useDeleteSetting } from '../../hooks/useAdminAPI';
import { Plus, Loader, AlertCircle, Trash2, Edit, Save, X } from 'lucide-react';
import { cn } from '../../lib/utils';

interface SettingInput {
  category: string;
  key: string;
  value: string;
  type: string;
  description: string;
  is_secret: boolean;
}

const Settings: React.FC = () => {
  const { hasPermission } = useAdmin();
  const [selectedCategory, setSelectedCategory] = useState<string>('general');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState<SettingInput>({
    category: 'general',
    key: '',
    value: '',
    type: 'string',
    description: '',
    is_secret: false,
  });

  const categories = ['general', 'security', 'email', 'notification', 'backup', 'api'];

  const { data: settingsData, isLoading } = useSettings(selectedCategory, 1, 100);
  const createMutation = useCreateSetting();
  const updateMutation = useUpdateSetting(editingId || '');
  const deleteMutation = useDeleteSetting();

  if (!hasPermission('settings:read')) {
    return (
      <div className="p-6 bg-red-50 rounded-lg border border-red-200">
        <p className="text-red-800">You don't have permission to view settings.</p>
      </div>
    );
  }

  const handleSaveSetting = async () => {
    if (!formData.key) {
      alert('Setting key is required');
      return;
    }

    try {
      if (editingId) {
        await updateMutation.mutateAsync({
          value: formData.type === 'number' ? parseFloat(formData.value) : 
                  formData.type === 'boolean' ? formData.value === 'true' : formData.value,
          description: formData.description || undefined,
          is_secret: formData.is_secret,
        });
      } else {
        await createMutation.mutateAsync({
          category: formData.category,
          key: formData.key,
          value: formData.type === 'number' ? parseFloat(formData.value) : 
                 formData.type === 'boolean' ? formData.value === 'true' : formData.value,
          type: formData.type,
          description: formData.description || undefined,
          is_secret: formData.is_secret,
        });
      }

      setFormData({
        category: 'general',
        key: '',
        value: '',
        type: 'string',
        description: '',
        is_secret: false,
      });
      setShowCreateForm(false);
      setEditingId(null);
    } catch (error) {
      alert(`Error saving setting: ${error}`);
    }
  };

  const handleDeleteSetting = async (settingId: string) => {
    if (window.confirm('Are you sure you want to delete this setting?')) {
      try {
        await deleteMutation.mutateAsync(settingId);
      } catch (error) {
        alert(`Error deleting setting: ${error}`);
      }
    }
  };

  const renderSettingValue = (setting: any) => {
    if (setting.is_secret) {
      return <span className="text-gray-500">***REDACTED***</span>;
    }

    if (setting.type === 'boolean') {
      return <span className="font-semibold">{String(setting.value)}</span>;
    }

    if (typeof setting.value === 'object') {
      return <code className="bg-gray-100 px-2 py-1 rounded text-xs">{JSON.stringify(setting.value)}</code>;
    }

    return <span>{String(setting.value)}</span>;
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-600 mt-1">Configure system settings and options</p>
        </div>
        {hasPermission('settings:write') && (
          <button
            onClick={() => {
              setShowCreateForm(!showCreateForm);
              setEditingId(null);
            }}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Plus className="w-5 h-5" />
            <span>Add Setting</span>
          </button>
        )}
      </div>

      {/* Create/Edit Form */}
      {(showCreateForm || editingId) && hasPermission('settings:write') && (
        <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">
            {editingId ? 'Edit Setting' : 'Add New Setting'}
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            {!editingId && (
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Category *
                </label>
                <select
                  value={formData.category}
                  onChange={(e) => setFormData((prev) => ({ ...prev, category: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat.charAt(0).toUpperCase() + cat.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Key {!editingId && '*'}
              </label>
              <input
                type="text"
                value={formData.key}
                onChange={(e) => setFormData((prev) => ({ ...prev, key: e.target.value }))}
                placeholder="setting_key"
                disabled={!!editingId}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">Type</label>
              <select
                value={formData.type}
                onChange={(e) => setFormData((prev) => ({ ...prev, type: e.target.value }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="string">String</option>
                <option value="number">Number</option>
                <option value="boolean">Boolean</option>
                <option value="json">JSON</option>
              </select>
            </div>

            <div>
              <label className="flex items-center space-x-2 mt-6">
                <input
                  type="checkbox"
                  checked={formData.is_secret}
                  onChange={(e) => setFormData((prev) => ({ ...prev, is_secret: e.target.checked }))}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700">Secret (mask in UI)</span>
              </label>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">Value</label>
            {formData.type === 'boolean' ? (
              <select
                value={formData.value}
                onChange={(e) => setFormData((prev) => ({ ...prev, value: e.target.value }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select...</option>
                <option value="true">True</option>
                <option value="false">False</option>
              </select>
            ) : (
              <textarea
                value={formData.value}
                onChange={(e) => setFormData((prev) => ({ ...prev, value: e.target.value }))}
                placeholder="Setting value..."
                rows={formData.type === 'json' ? 4 : 2}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
              />
            )}
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">Description</label>
            <input
              type="text"
              value={formData.description}
              onChange={(e) => setFormData((prev) => ({ ...prev, description: e.target.value }))}
              placeholder="Brief description of this setting..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div className="flex space-x-3">
            <button
              onClick={handleSaveSetting}
              disabled={(createMutation.isPending || updateMutation.isPending) && !formData.key}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              <Save className="w-4 h-4" />
              <span>{editingId ? 'Update' : 'Create'} Setting</span>
            </button>
            <button
              onClick={() => {
                setShowCreateForm(false);
                setEditingId(null);
              }}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Category Tabs */}
      <div className="flex space-x-2 border-b border-gray-200 overflow-x-auto">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={cn(
              'px-4 py-2 font-medium text-sm whitespace-nowrap border-b-2 transition-colors',
              selectedCategory === cat
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-600 hover:text-gray-900'
            )}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      {/* Settings Table */}
      <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center h-96">
            <Loader className="w-8 h-8 animate-spin text-blue-600" />
          </div>
        ) : !settingsData || settingsData.items.length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-600">No settings in {selectedCategory} category</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Key</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Value</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Description</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Type</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Actions</th>
                </tr>
              </thead>
              <tbody>
                {settingsData.items.map((setting) => (
                  <tr key={setting.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium text-gray-900">{setting.key}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {renderSettingValue(setting)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {setting.description || '-'}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span className="inline-block px-2 py-1 rounded text-xs bg-gray-100 text-gray-800">
                        {setting.type || 'string'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm space-x-2">
                      {hasPermission('settings:write') && (
                        <button
                          onClick={() => {
                            setEditingId(setting.id);
                            setFormData({
                              category: setting.category,
                              key: setting.key,
                              value: String(setting.value || ''),
                              type: setting.type || 'string',
                              description: setting.description || '',
                              is_secret: setting.is_secret,
                            });
                            setShowCreateForm(false);
                          }}
                          className="p-2 hover:bg-gray-100 rounded-lg text-gray-600"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                      )}
                      {hasPermission('settings:write') && (
                        <button
                          onClick={() => handleDeleteSetting(setting.id)}
                          className="p-2 hover:bg-red-50 rounded-lg text-red-600"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings;
