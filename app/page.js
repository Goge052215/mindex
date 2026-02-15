'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Bebas_Neue } from 'next/font/google';
import { useTheme } from './ThemeContext';

const logoFont = Bebas_Neue({ subsets: ['latin'], weight: '400' });

export default function SentimentAnalyzer() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState("");
  const [distribution, setDistribution] = useState(null);
  const [history, setHistory] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const { themeColor } = useTheme();

  const normalizeHistory = (entries) => {
    if (!Array.isArray(entries)) return [];
    return entries.map((entry) => {
      if (typeof entry === 'string') {
        const [rawText, rawSentiment] = entry.split('->');
        const text = (rawText || '').replace(/^"|"$/g, '').trim();
        const sentiment = (rawSentiment || '').trim() || 'Unknown';
        return {
          text,
          sentiment,
          distribution: null,
          createdAt: new Date().toISOString()
        };
      }
      return {
        text: entry.text || entry.input || '',
        sentiment: entry.sentiment || entry.result || 'Unknown',
        distribution: entry.distribution || null,
        createdAt: entry.createdAt || new Date().toISOString()
      };
    });
  };

  useEffect(() => {
    const savedHistory = localStorage.getItem('history');
    if (savedHistory) {
      setHistory(normalizeHistory(JSON.parse(savedHistory)));
    }
  }, []);

  // Save history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('history', JSON.stringify(history));
  }, [history]);

  const handleExecute = async () => {
    if (!input) return;

    const lines = input
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) return;

    try {
      const payload = lines.length > 1 ? lines : lines[0];
      const res = await fetch('/api', {
        method: 'POST',
        body: JSON.stringify({ expression: payload }),
      });
      const data = await res.json();

      if (data.status === "success") {
        if (Array.isArray(data.data)) {
          const createdAt = new Date().toISOString();
          const entries = data.data.map((entry) => ({
            text: entry.text,
            sentiment: entry.sentiment,
            distribution: entry.distribution,
            createdAt
          }));
          setBatchResults(entries);
          setResult("");
          setDistribution(null);
          setHistory((prev) => [...entries, ...prev].slice(0, 10));
        } else {
          const newEntry = {
            text: lines[0],
            sentiment: data.data,
            distribution: data.distribution,
            createdAt: new Date().toISOString()
          };
          setResult(data.data);
          setDistribution(data.distribution);
          setBatchResults([]);
          setHistory((prev) => [newEntry, ...prev].slice(0, 10));
        }
      } else {
        setResult("Error");
        setDistribution(null);
        setBatchResults(lines.length > 1 ? lines.map((line) => ({
          text: line,
          sentiment: "Error",
          distribution: null
        })) : []);
      }
    } catch {
      setResult("Error");
      setDistribution(null);
      setBatchResults(lines.length > 1 ? lines.map((line) => ({
        text: line,
        sentiment: "Error",
        distribution: null
      })) : []);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const exportJson = () => {
    if (history.length === 0) return;
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mindex-history.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportCsv = () => {
    if (history.length === 0) return;
    const escapeCsv = (value) => {
      const str = String(value ?? '');
      if (str.includes('"') || str.includes(',') || str.includes('\n')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    };

    const header = ['text', 'sentiment', 'positive', 'neutral', 'negative', 'createdAt'];
    const rows = history.map((entry) => {
      const dist = entry.distribution || {};
      return [
        escapeCsv(entry.text),
        escapeCsv(entry.sentiment),
        escapeCsv(dist.Positive ?? ''),
        escapeCsv(dist.Neutral ?? ''),
        escapeCsv(dist.Negative ?? ''),
        escapeCsv(entry.createdAt ?? '')
      ].join(',');
    });

    const csvContent = [header.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mindex-history.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const getThemeButtonClass = (color) => {
    const map = {
      orange: 'bg-orange-500',
      blue: 'bg-blue-500',
      green: 'bg-green-500',
      purple: 'bg-purple-500',
      red: 'bg-red-500',
      yellow: 'bg-yellow-500',
      pink: 'bg-pink-500',
      teal: 'bg-teal-500',
    };
    return map[color] || 'bg-orange-500';
  };

  const getThemeHoverClass = (color) => {
    const map = {
      orange: 'hover:bg-orange-600',
      blue: 'hover:bg-blue-600',
      green: 'hover:bg-green-600',
      purple: 'hover:bg-purple-600',
      red: 'hover:bg-red-600',
      yellow: 'hover:bg-yellow-600',
      pink: 'hover:bg-pink-600',
      teal: 'hover:bg-teal-600',
    };
    return map[color] || 'hover:bg-orange-600';
  };

  const themeClass = getThemeButtonClass(themeColor);
  const historyStats = history.reduce((acc, entry) => {
    const label = entry.sentiment || 'Unknown';
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});
  const trend = history.slice(0, 10);
  const sentimentColors = {
    Positive: 'bg-green-500',
    Negative: 'bg-red-500',
    Neutral: 'bg-gray-500',
    Error: 'bg-yellow-500',
    Unknown: 'bg-gray-400'
  };
  return (
    <div className="flex flex-col items-center justify-center p-10 bg-gray-100 dark:bg-black min-h-screen font-sans transition-colors duration-300">
      <div className="flex justify-between w-full max-w-4xl mb-8 items-center">
        <h1 className={`${logoFont.className} text-4xl font-bold text-gray-800 dark:text-white tracking-wider`}>Mindex</h1>
        <Link 
          href="/settings" 
          className={`px-4 py-2 text-white rounded transition ${themeClass} ${getThemeHoverClass(themeColor)}`}
        >
          Settings
        </Link>
      </div>

      <div className="flex justify-between w-full max-w-4xl mb-8 items-center">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white">A handy, mini tool that analyzes your sentiment</h2>
      </div>

      <div className="flex flex-col md:flex-row gap-8 w-full max-w-4xl">
        {/* Input sections */}
        <div className="bg-white dark:bg-black p-6 rounded-xl shadow-xl border-4 border-gray-300 dark:border-gray-700 flex-1 transition-colors duration-300">
          <div className="mb-4">
             <label htmlFor="sentiment-input" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
               Enter text for analysis
             </label>
             <textarea
               id="sentiment-input"
               className="w-full p-4 bg-gray-50 dark:bg-black text-gray-900 dark:text-white rounded-lg border border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-h-[150px] resize-none"
               placeholder="Type something here (e.g., 'I love this product!')..."
               value={input}
               onChange={(e) => setInput(e.target.value)}
             />
          </div>

          {result && (
             <div className="mb-6 p-4 bg-gray-100 dark:bg-black rounded-lg text-center">
                <span className="text-gray-500 dark:text-gray-400 text-sm uppercase tracking-wide">Prediction</span>
                <div className={`text-4xl font-bold mt-2 mb-4 ${
                  result === 'Positive' ? 'text-green-500' :
                  result === 'Negative' ? 'text-red-500' :
                  'text-gray-700 dark:text-gray-200'
                }`}>
                  {result}
                </div>
                
                {distribution && (
                  <div className="grid grid-cols-3 gap-4 border-t border-gray-200 dark:border-gray-700 pt-4">
                    {Object.entries(distribution).map(([label, score]) => (
                      <div key={label} className="flex flex-col items-center gap-1">
                        <span className={`text-sm font-bold ${
                          label === 'Positive' ? 'text-green-500' :
                          label === 'Negative' ? 'text-red-500' :
                          'text-gray-500 dark:text-gray-400'
                        }`}>
                          {score}%
                        </span>
                        <div className="w-full h-2 bg-gray-200 dark:bg-gray-800 rounded">
                          <div
                            className={`h-2 rounded ${sentimentColors[label] || 'bg-gray-400'}`}
                            style={{ width: `${score}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-400 uppercase">{label}</span>
                      </div>
                    ))}
                  </div>
                )}
             </div>
          )}

          {batchResults.length > 0 && (
            <div className="mb-6 p-4 bg-gray-100 dark:bg-black rounded-lg">
              <div className="text-gray-500 dark:text-gray-400 text-sm uppercase tracking-wide mb-3">
                Batch Results
              </div>
              <div className="space-y-3">
                {batchResults.map((entry, idx) => (
                  <div key={`${entry.text}-${idx}`} className="p-3 bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded">
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-gray-700 dark:text-gray-300 break-all">
                        {entry.text}
                      </div>
                      <div className={`text-sm font-bold ${
                        entry.sentiment === 'Positive' ? 'text-green-500' :
                        entry.sentiment === 'Negative' ? 'text-red-500' :
                        entry.sentiment === 'Error' ? 'text-yellow-500' :
                        'text-gray-500 dark:text-gray-400'
                      }`}>
                        {entry.sentiment}
                      </div>
                    </div>
                    {entry.distribution && (
                      <div className="grid grid-cols-3 gap-3 mt-3">
                        {Object.entries(entry.distribution).map(([label, score]) => (
                          <div key={label} className="flex flex-col gap-1">
                            <span className="text-xs text-gray-400 uppercase">{label}</span>
                            <div className="w-full h-2 bg-gray-200 dark:bg-gray-800 rounded">
                              <div
                                className={`h-2 rounded ${sentimentColors[label] || 'bg-gray-400'}`}
                                style={{ width: `${score}%` }}
                              />
                            </div>
                            <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">{score}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-4">
            <button
              onClick={() => {
                setInput("");
                setResult("");
                setDistribution(null);
              }}
              className="px-6 py-3 bg-gray-200 dark:bg-black text-gray-700 dark:text-white rounded-lg hover:bg-gray-300 dark:hover:bg-gray-900 transition-colors font-semibold"
            >
              Clear
            </button>
            <button
              onClick={handleExecute}
              className={`flex-1 px-6 py-3 text-white rounded-lg transition-colors font-bold text-lg shadow-lg ${themeClass} ${getThemeHoverClass(themeColor)}`}
            >
              Analyze Sentiment
            </button>
          </div>
        </div>

        {/* History Section */}
        <div className="bg-white dark:bg-black p-6 rounded-xl shadow-xl border-4 border-gray-300 dark:border-gray-700 w-full md:w-72 h-fit transition-colors duration-300">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-700 dark:text-gray-200">History</h2>
            {history.length > 0 && (
              <button onClick={clearHistory} className="text-xs text-red-500 hover:underline">
                Clear
              </button>
            )}
          </div>
          <div className="mb-4 space-y-3">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase">Summary</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {Object.keys(historyStats).length === 0 && (
                <div className="text-gray-400 dark:text-gray-500">No data</div>
              )}
              {Object.entries(historyStats).map(([label, count]) => (
                <div key={label} className="flex items-center gap-2">
                  <span className={`inline-block w-2 h-2 rounded-full ${sentimentColors[label] || 'bg-gray-400'}`} />
                  <span className="text-gray-600 dark:text-gray-300">{label}</span>
                  <span className="text-gray-500 dark:text-gray-400">({count})</span>
                </div>
              ))}
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 uppercase mb-2">Trend</div>
              <div className="flex flex-wrap gap-1">
                {trend.length === 0 ? (
                  <span className="text-gray-400 dark:text-gray-500 text-xs">No data</span>
                ) : (
                  trend.map((entry, idx) => (
                    <span
                      key={`${entry.text}-${idx}`}
                      className={`inline-block w-3 h-3 rounded-full ${sentimentColors[entry.sentiment] || 'bg-gray-400'}`}
                      title={entry.sentiment}
                    />
                  ))
                )}
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={exportJson}
                className="flex-1 px-3 py-2 text-xs text-white rounded-lg transition-colors font-semibold bg-gray-700 hover:bg-gray-800"
              >
                Export JSON
              </button>
              <button
                onClick={exportCsv}
                className="flex-1 px-3 py-2 text-xs text-white rounded-lg transition-colors font-semibold bg-gray-700 hover:bg-gray-800"
              >
                Export CSV
              </button>
            </div>
          </div>
          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300 max-h-[400px] overflow-y-auto">
            {history.length === 0 ? (
              <li className="italic text-gray-400 dark:text-gray-500">No history yet</li>
            ) : (
              history.map((entry, idx) => {
                const displayText = entry.text || '';
                return (
                  <li key={`${displayText}-${idx}`} className="p-2 bg-gray-50 dark:bg-black rounded border border-gray-100 dark:border-gray-600 font-mono break-all transition-colors duration-200">
                    <div className="text-xs text-gray-400 dark:text-gray-500">{entry.sentiment}</div>
                    <div>{displayText}</div>
                  </li>
                );
              })
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}
