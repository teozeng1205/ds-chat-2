import { CHANGELOG } from "../lib/changelog";

interface Props { onClose: () => void; }

export function ChangelogModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
         onClick={onClose}>
      <div className="bg-white dark:bg-slate-900 rounded-xl shadow-xl w-full max-w-md mx-4 p-6"
           onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-slate-800 dark:text-slate-100">What's New</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-lg leading-none">×</button>
        </div>
        <div className="space-y-5 max-h-80 overflow-y-auto pr-1">
          {CHANGELOG.map(entry => (
            <div key={entry.version}>
              <p className="text-xs font-medium text-slate-400 mb-1">{entry.date}</p>
              <ul className="space-y-1">
                {entry.items.map((item, i) => (
                  <li key={i} className="text-sm text-slate-700 dark:text-slate-300 flex gap-2">
                    <span className="text-blue-500 mt-0.5">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
