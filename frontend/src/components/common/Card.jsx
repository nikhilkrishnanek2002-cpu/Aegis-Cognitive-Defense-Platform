export function Card({ title, subtitle, children, className = '', action = null }) {
  return (
    <div className={`bg-slate-800/50 border border-slate-700 rounded-lg p-4 ${className}`}>
      <div className="flex items-start justify-between mb-4">
        <div>
          {title && <h3 className="text-lg font-semibold text-white">{title}</h3>}
          {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
        </div>
        {action && <div className="flex-shrink-0">{action}</div>}
      </div>
      <div>{children}</div>
    </div>
  )
}

export default Card
