// app/settings/page.js
'use client';
import Link from 'next/link';
import { useTheme } from '../ThemeContext';

export default function Settings() {
  const { themeColor, changeTheme, themeMode, toggleMode } = useTheme();

  const colors = [
    { name: 'Orange', value: 'orange', class: 'bg-orange-500' },
    { name: 'Blue', value: 'blue', class: 'bg-blue-500' },
    { name: 'Green', value: 'green', class: 'bg-green-500' },
    { name: 'Purple', value: 'purple', class: 'bg-purple-500' },
    { name: 'Red', value: 'red', class: 'bg-red-500' },
  ];

  return (
    <div className="flex flex-col items-center justify-center p-10 bg-gray-100 dark:bg-black min-h-screen font-sans transition-colors duration-300">
      <div className="w-full max-w-2xl bg-white dark:bg-black p-8 rounded-xl shadow-xl border-4 border-gray-300 dark:border-gray-700 transition-colors duration-300">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Settings</h1>
          <Link href="/" className="text-blue-600 dark:text-blue-400 hover:underline">
            &larr; Back to the main page
          </Link>
        </div>

        <div className="mb-8 border-b border-gray-200 dark:border-gray-700 pb-8">
          <h2 className="text-xl font-bold mb-4 text-gray-700 dark:text-gray-200">Appearance</h2>
          
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-gray-700 dark:text-gray-300 font-medium">Dark Mode</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">Toggle dark/light theme</p>
            </div>
            <button
              onClick={toggleMode}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                themeMode === 'dark' ? 'bg-blue-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  themeMode === 'dark' ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>

        <div className="mb-8">
          <h2 className="text-xl font-bold mb-4 text-gray-700 dark:text-gray-200">Theme Color</h2>
          <p className="text-gray-500 dark:text-gray-400 mb-4">Choose a color for the buttons and highlights.</p>
          
          <div className="flex flex-wrap gap-4">
            {colors.map((color) => (
              <button
                key={color.value}
                onClick={() => changeTheme(color.value)}
                className={`w-16 h-16 rounded-full ${color.class} ${
                  themeColor === color.value ? 'ring-4 ring-offset-2 ring-gray-400 dark:ring-gray-500' : ''
                } transition-all transform hover:scale-110`}
                aria-label={`Select ${color.name} theme`}
              />
            ))}
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 pt-8">
          <h2 className="text-xl font-bold mb-4 text-gray-700 dark:text-gray-200">Data Management</h2>
          <button
            onClick={() => {
              if (confirm('Are you sure you want to clear all history?')) {
                localStorage.removeItem('history');
                alert('History cleared!');
              }
            }}
            className="px-4 py-2 bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-200 rounded hover:bg-red-200 dark:hover:bg-red-800 transition font-bold"
          >
            Clear All History Data
          </button>
        </div>
      </div>
    </div>
  );
}
