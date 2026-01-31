/**
 * Dashboard Layout
 * ================
 * 
 * Main application layout with sidebar navigation.
 */

import type { ComponentType, SVGProps } from 'react';
import { useState, Fragment } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { Dialog, Transition, Menu } from '@headlessui/react';
import {
  Bars3Icon,
  XMarkIcon,
  HomeIcon,
  CircleStackIcon,
  UserGroupIcon,
  UsersIcon,
  ClipboardDocumentListIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
  ChevronDownIcon,
  BuildingOfficeIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { useAuthStore, Organization } from '@/stores/auth';

type NavItem = {
  name: string;
  href: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
};

/**
 * Sidebar navigation component
 */
function Sidebar({ mobile = false, onClose }: { mobile?: boolean; onClose?: () => void }) {
  const navigate = useNavigate();
  const { user, currentOrg, availableOrgs, switchOrg, logout } = useAuthStore();

  const roles = user?.roles ?? [];
  const isAdmin = roles.includes('org_admin') || roles.includes('system_admin');
  const isReviewer = roles.includes('knowledge_reviewer') || isAdmin;

  const navigation: NavItem[] = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'Memories', href: '/memories', icon: CircleStackIcon },
    ...(isReviewer ? [{ name: 'Review Queue', href: '/review', icon: ClipboardDocumentListIcon }] : []),
    { name: 'Teams', href: '/teams', icon: UserGroupIcon },
    ...(isAdmin ? [{ name: 'Users', href: '/users', icon: UsersIcon }] : []),
    ...(isAdmin ? [{ name: 'Audit Log', href: '/audit', icon: ClipboardDocumentListIcon }] : []),
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
  ];

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleOrgSwitch = (org: Organization) => {
    switchOrg(org);
    // Could refresh data here or trigger API call
    window.location.reload();
  };

  return (
    <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-gray-900 px-6 pb-4">
      {/* Logo */}
      <div className="flex h-16 shrink-0 items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary-500 rounded-lg flex items-center justify-center">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>
          <span className="text-xl font-bold text-white">Ninai</span>
        </div>
      </div>

      {/* Organization Switcher */}
      <Menu as="div" className="relative">
        <Menu.Button className="flex w-full items-center gap-x-3 rounded-lg bg-gray-800 p-3 text-sm font-semibold leading-6 text-white hover:bg-gray-700">
          <BuildingOfficeIcon className="h-5 w-5 text-gray-400" />
          <span className="flex-1 text-left truncate">
            {currentOrg?.name || 'Select Organization'}
          </span>
          <ChevronDownIcon className="h-4 w-4 text-gray-400" />
        </Menu.Button>

        <Transition
          as={Fragment}
          enter="transition ease-out duration-100"
          enterFrom="transform opacity-0 scale-95"
          enterTo="transform opacity-100 scale-100"
          leave="transition ease-in duration-75"
          leaveFrom="transform opacity-100 scale-100"
          leaveTo="transform opacity-0 scale-95"
        >
          <Menu.Items className="absolute left-0 right-0 z-10 mt-2 origin-top-left rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
            <div className="py-1">
              {availableOrgs.map((org) => (
                <Menu.Item key={org.id}>
                  {({ active }) => (
                    <button
                      onClick={() => handleOrgSwitch(org)}
                      className={clsx(
                        'w-full text-left px-4 py-2 text-sm',
                        active ? 'bg-gray-100' : '',
                        org.id === currentOrg?.id
                          ? 'text-primary-600 font-medium'
                          : 'text-gray-700'
                      )}
                    >
                      {org.name}
                      {org.id === currentOrg?.id && (
                        <span className="ml-2 text-xs text-gray-400">Current</span>
                      )}
                    </button>
                  )}
                </Menu.Item>
              ))}
            </div>
          </Menu.Items>
        </Transition>
      </Menu>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col">
        <ul role="list" className="flex flex-1 flex-col gap-y-7">
          <li>
            <ul role="list" className="-mx-2 space-y-1">
              {navigation.map((item) => (
                <li key={item.name}>
                  <NavLink
                    to={item.href}
                    onClick={mobile ? onClose : undefined}
                    className={({ isActive }) =>
                      clsx(
                        'group flex gap-x-3 rounded-lg p-2 text-sm font-semibold leading-6',
                        isActive
                          ? 'bg-gray-800 text-white'
                          : 'text-gray-400 hover:text-white hover:bg-gray-800'
                      )
                    }
                  >
                    <item.icon className="h-6 w-6 shrink-0" aria-hidden="true" />
                    {item.name}
                  </NavLink>
                </li>
              ))}
            </ul>
          </li>

          {/* User section */}
          <li className="mt-auto">
            <div className="flex items-center gap-x-4 px-2 py-3 text-sm font-semibold leading-6 text-white">
              <div className="h-8 w-8 rounded-full bg-gray-700 flex items-center justify-center">
                {user?.avatar_url ? (
                  <img
                    src={user.avatar_url}
                    alt=""
                    className="h-8 w-8 rounded-full"
                  />
                ) : (
                  <span className="text-sm">
                    {user?.display_name?.charAt(0).toUpperCase() || 'U'}
                  </span>
                )}
              </div>
              <div className="flex-1 truncate">
                <p className="truncate">{user?.display_name}</p>
                <p className="text-xs text-gray-400 truncate">{user?.email}</p>
              </div>
              <button
                onClick={handleLogout}
                className="p-1 text-gray-400 hover:text-white"
                title="Sign out"
              >
                <ArrowRightOnRectangleIcon className="h-5 w-5" />
              </button>
            </div>
          </li>
        </ul>
      </nav>
    </div>
  );
}

/**
 * Dashboard Layout Component
 */
export function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <>
      {/* Mobile sidebar */}
      <Transition.Root show={sidebarOpen} as={Fragment}>
        <Dialog as="div" className="relative z-50 lg:hidden" onClose={setSidebarOpen}>
          <Transition.Child
            as={Fragment}
            enter="transition-opacity ease-linear duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="transition-opacity ease-linear duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-900/80" />
          </Transition.Child>

          <div className="fixed inset-0 flex">
            <Transition.Child
              as={Fragment}
              enter="transition ease-in-out duration-300 transform"
              enterFrom="-translate-x-full"
              enterTo="translate-x-0"
              leave="transition ease-in-out duration-300 transform"
              leaveFrom="translate-x-0"
              leaveTo="-translate-x-full"
            >
              <Dialog.Panel className="relative mr-16 flex w-full max-w-xs flex-1">
                <Transition.Child
                  as={Fragment}
                  enter="ease-in-out duration-300"
                  enterFrom="opacity-0"
                  enterTo="opacity-100"
                  leave="ease-in-out duration-300"
                  leaveFrom="opacity-100"
                  leaveTo="opacity-0"
                >
                  <div className="absolute left-full top-0 flex w-16 justify-center pt-5">
                    <button
                      type="button"
                      className="-m-2.5 p-2.5"
                      onClick={() => setSidebarOpen(false)}
                    >
                      <span className="sr-only">Close sidebar</span>
                      <XMarkIcon className="h-6 w-6 text-white" aria-hidden="true" />
                    </button>
                  </div>
                </Transition.Child>
                <Sidebar mobile onClose={() => setSidebarOpen(false)} />
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </Dialog>
      </Transition.Root>

      {/* Static sidebar for desktop */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
        <Sidebar />
      </div>

      {/* Main content area */}
      <div className="lg:pl-72">
        {/* Mobile header */}
        <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8 lg:hidden">
          <button
            type="button"
            className="-m-2.5 p-2.5 text-gray-700 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <span className="sr-only">Open sidebar</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>
          
          <div className="flex-1 text-sm font-semibold leading-6 text-gray-900">
            Ninai
          </div>
        </div>

        {/* Page content */}
        <main className="py-6">
          <div className="px-4 sm:px-6 lg:px-8 max-w-full">
            <Outlet />
          </div>
        </main>
      </div>
    </>
  );
}
