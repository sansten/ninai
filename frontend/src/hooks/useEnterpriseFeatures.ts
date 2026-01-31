import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';

interface FeatureFlags {
  admin_operations: boolean;
  drift_detection: boolean;
  auto_eval_benchmarks: boolean;
  memory_observability: boolean;
}

/**
 * Hook to check which enterprise features are available.
 * 
 * Returns feature flags based on license status:
 * - Community build: All flags false
 * - Enterprise build with valid license: Flags true
 * 
 * Use this to conditionally render enterprise UI components.
 */
export function useEnterpriseFeatures() {
  const { data: features, isLoading } = useQuery({
    queryKey: ['enterprise-features'],
    queryFn: async () => {
      try {
        const response = await apiClient.get<FeatureFlags>('/features');
        return response.data;
      } catch (error) {
        // If endpoint doesn't exist (404), assume community build
        return {
          admin_operations: false,
          drift_detection: false,
          auto_eval_benchmarks: false,
          memory_observability: false,
        } as FeatureFlags;
      }
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retry: false, // Don't retry on 404
  });

  return {
    features: features ?? {
      admin_operations: false,
      drift_detection: false,
      auto_eval_benchmarks: false,
      memory_observability: false,
    },
    isLoading,
    hasAdminOperations: features?.admin_operations ?? false,
    hasDriftDetection: features?.drift_detection ?? false,
    hasAutoEvalBenchmarks: features?.auto_eval_benchmarks ?? false,
    hasMemoryObservability: features?.memory_observability ?? false,
  };
}
