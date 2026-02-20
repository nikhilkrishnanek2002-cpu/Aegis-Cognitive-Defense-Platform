import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'

export function DashboardLayout({ children }) {
  return (
    <div className="flex h-screen bg-slate-900">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Topbar */}
        <Topbar />

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto bg-slate-900">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

export default DashboardLayout
