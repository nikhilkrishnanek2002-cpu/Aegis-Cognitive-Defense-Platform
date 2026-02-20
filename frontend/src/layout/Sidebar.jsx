export function Sidebar() {
  const navItems = [
    { name: 'Dashboard', icon: '📊', href: '/' },
    { name: 'Radar Live', icon: '📡', href: '/radar' },
    { name: 'Threat Analysis', icon: '🚨', href: '/threats' },
    { name: 'EW Control', icon: '⚙️', href: '/ew' },
    { name: 'Model Monitor', icon: '🤖', href: '/monitor' },
    { name: 'Settings', icon: '⚡', href: '/settings' },
  ]

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-700 h-screen sticky top-0 flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-slate-700">
        <h1 className="text-xl font-bold text-cyan-400">⚔️ AEGIS</h1>
        <p className="text-xs text-slate-400">Defense Monitoring</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {navItems.map((item) => (
          <a
            key={item.href}
            href={item.href}
            className="flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors hover:bg-slate-800 text-slate-300 hover:text-cyan-400"
          >
            <span className="text-lg">{item.icon}</span>
            <span className="text-sm font-medium">{item.name}</span>
          </a>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700 space-y-2">
        <div className="text-xs text-slate-500">
          <p>System v2.0</p>
          <p>© 2026 Aegis</p>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
