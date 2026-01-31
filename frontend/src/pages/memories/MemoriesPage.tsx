/**
 * Memories Page
 * =============
 * 
 * Memory listing with search and filters.
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { apiClient } from '@/lib/api';
import { useCurrentOrg } from '@/stores/auth';
import type { Memory, MemorySearchResult, MemoryType, VisibilityLevel } from '@/types/api';

// Memory type colors
const typeColors: Record<MemoryType, string> = {
  episodic: 'badge-primary',
  semantic: 'badge-success',
  procedural: 'badge-warning',
  working: 'badge-gray',
  strategic: 'badge-danger',
  context: 'badge-primary',
};

// Visibility colors
const visibilityColors: Record<VisibilityLevel, string> = {
  private: 'badge-gray',
  team: 'badge-primary',
  department: 'badge-warning',
  organization: 'badge-success',
};

/**
 * Memory Card Component
 */
interface MemoryCardProps {
  memory: Memory;
  score?: number;
}

function MemoryCard({ memory, score }: MemoryCardProps) {
  return (
    <Link
      to={`/memories/${memory.id}`}
      className="card hover:shadow-md transition-shadow block"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <h3 className="text-base font-semibold text-gray-900 truncate">
            {memory.title || 'Untitled Memory'}
          </h3>
          <p className="mt-1 text-sm text-gray-500 line-clamp-2">
            {memory.summary || memory.content.slice(0, 150)}
          </p>
        </div>
        {score !== undefined && (
          <div className="ml-4 text-xs text-gray-400">
            {Math.round(score * 100)}% match
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center gap-2 flex-wrap">
        <span className={typeColors[memory.memory_type]}>
          {memory.memory_type}
        </span>
        <span className={visibilityColors[memory.visibility_level]}>
          {memory.visibility_level}
        </span>
        {memory.tags?.slice(0, 3).map((tag) => (
          <span key={tag} className="badge-gray">
            {tag}
          </span>
        ))}
        {memory.tags && memory.tags.length > 3 && (
          <span className="text-xs text-gray-400">
            +{memory.tags.length - 3} more
          </span>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-gray-400">
        <span>Importance: {memory.importance_score}</span>
        <span>{new Date(memory.created_at).toLocaleDateString()}</span>
      </div>
    </Link>
  );
}

/**
 * Memories Page Component
 */
export function MemoriesPage() {
  const org = useCurrentOrg();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<MemoryType | ''>('');
  const [selectedVisibility, setSelectedVisibility] = useState<VisibilityLevel | ''>('');
  const [showFilters, setShowFilters] = useState(false);

  // Search query
  const { data: searchResults, isLoading } = useQuery<{ items: MemorySearchResult[] }>({
    queryKey: ['memories', org.id, searchQuery, selectedType, selectedVisibility],
    queryFn: async () => {
      const params: Record<string, unknown> = {
        query: searchQuery || '*',
        limit: 50,
      };
      if (selectedType) params.memory_types = [selectedType];
      if (selectedVisibility) params.visibility_levels = [selectedVisibility];

      const response = await apiClient.post('/memories/search', params);
      return response.data;
    },
    staleTime: 1000 * 30, // 30 seconds
  });

  const memoryTypes: MemoryType[] = ['episodic', 'semantic', 'procedural', 'working', 'strategic', 'context'];
  const visibilityLevels: VisibilityLevel[] = ['private', 'team', 'department', 'organization'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Memories</h1>
          <p className="text-gray-500 mt-1">
            Search and manage your AI memory store
          </p>
        </div>
        <button className="btn-primary">
          <PlusIcon className="h-5 w-5 mr-2" />
          New Memory
        </button>
      </div>

      {/* Search & Filters */}
      <div className="card">
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search memories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input pl-10"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={clsx(
              'btn-secondary',
              showFilters && 'bg-gray-300'
            )}
          >
            <FunnelIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Filter Panel */}
        {showFilters && (
          <div className="mt-4 pt-4 border-t border-gray-200 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="label">Memory Type</label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value as MemoryType | '')}
                className="input"
              >
                <option value="">All types</option>
                {memoryTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Visibility</label>
              <select
                value={selectedVisibility}
                onChange={(e) => setSelectedVisibility(e.target.value as VisibilityLevel | '')}
                className="input"
              >
                <option value="">All visibility levels</option>
                {visibilityLevels.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      ) : searchResults?.items?.length ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {searchResults.items.map(({ memory, score }) => (
            <MemoryCard key={memory.id} memory={memory} score={score} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <CircleStackIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-semibold text-gray-900">No memories found</h3>
          <p className="mt-1 text-sm text-gray-500">
            Try adjusting your search or filters.
          </p>
        </div>
      )}
    </div>
  );
}

// Import for empty state
import { CircleStackIcon } from '@heroicons/react/24/outline';
