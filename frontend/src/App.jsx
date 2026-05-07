import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

/* ── Constants ────────────────────────────────────────────────────────────── */
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const SUGGESTIONS = [
  { icon: '📅', text: 'How many leave days do I have left?' },
  { icon: '🏠', text: 'Am I eligible for remote work?' },
  { icon: '📊', text: 'What salary increase can I expect based on my rating?' },
  { icon: '📚', text: 'How much training budget do I have remaining?' },
]

const SOURCE_LABELS = {
  rag: 'Source: Corporate Policies',
  structured_data: 'Source: Employee Database',
  both: 'Source: Database & Policies',
  unknown: 'Source: General Knowledge',
}

/* ── App Component ────────────────────────────────────────────────────────── */
function App() {
  const [employeeId, setEmployeeId] = useState('EMP001')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  /* Auto-scroll to bottom on new messages */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  /* Focus input on mount */
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  /* ── Send message ──────────────────────────────────────────────────────── */
  const sendMessage = useCallback(async (text) => {
    const question = text || input.trim()
    if (!question || isLoading) return
    if (!employeeId.trim()) {
      setError('Please enter your Employee ID first')
      return
    }

    setError(null)
    setInput('')

    // Add user message
    const userMsg = { role: 'user', content: question, timestamp: new Date() }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          employee_id: employeeId.trim(),
          question,
          session_id: sessionId,
        }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error (${res.status})`)
      }

      const data = await res.json()

      // Update session ID from response
      if (data.session_id) setSessionId(data.session_id)

      // Add assistant message
      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        source: data.source,
        references: data.references || [],
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${err.message}. Please make sure the backend server is running.`,
        source: 'unknown',
        timestamp: new Date(),
        isError: true,
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }, [input, isLoading, employeeId, sessionId])

  /* ── Handle keyboard ───────────────────────────────────────────────────── */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  /* ── New chat ──────────────────────────────────────────────────────────── */
  const startNewChat = () => {
    setMessages([])
    setSessionId(null)
    setError(null)
    inputRef.current?.focus()
  }

  /* ── Render ────────────────────────────────────────────────────────────── */
  return (
    <div className="app">
      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">🤖</div>
            <h1>HR Assistant</h1>
          </div>
          <div className="sidebar-subtitle">AlNoor Technologies</div>
        </div>

        <div className="sidebar-config">
          <div className="config-section">
            <label className="config-label">Employee ID</label>
            <input
              className="config-input"
              type="text"
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value.toUpperCase())}
              placeholder="e.g. EMP001"
              spellCheck={false}
            />
          </div>

          {employeeId && (
            <div className="config-section">
              <label className="config-label">Quick Info</label>
              <div className="employee-card">
                <div className="employee-card-name">
                  Employee {employeeId}
                </div>
                <div className="employee-card-row">
                  <span className="employee-card-label">Status</span>
                  <span className="employee-card-value">Connected</span>
                </div>
                <div className="employee-card-row">
                  <span className="employee-card-label">Session</span>
                  <span className="employee-card-value">
                    {sessionId ? `#${sessionId.slice(0, 8)}` : 'New'}
                  </span>
                </div>
                <div className="employee-card-row">
                  <span className="employee-card-label">Messages</span>
                  <span className="employee-card-value">{messages.length}</span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="config-section">
              <div className="employee-card error">
                <div className="employee-card-name" style={{ color: '#ef4444' }}>
                  ⚠ {error}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <button className="btn-new-chat" onClick={startNewChat}>
            ✨ New Conversation
          </button>
        </div>
      </aside>

      {/* ── Main Chat Area ───────────────────────────────────────────────── */}
      <main className="main">
        <div className="topbar">
          <span className="topbar-title">
            {messages.length > 0 ? 'Conversation' : 'New Chat'}
          </span>
          {sessionId && (
            <span className="topbar-session">
              Session {sessionId.slice(0, 8)}
            </span>
          )}
        </div>

        <div className="chat-messages">
          {messages.length === 0 && !isLoading ? (
            /* ── Welcome Screen ────────────────────────────────────────── */
            <div className="welcome">
              <div className="welcome-icon">🤖</div>
              <h2>HR AI Assistant</h2>
              <p>
                Ask me about company policies, your leave balance,
                remote work eligibility, performance reviews, and more.
              </p>
              <div className="welcome-suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <button
                    key={i}
                    className="suggestion-card"
                    onClick={() => sendMessage(s.text)}
                  >
                    <div className="suggestion-card-icon">{s.icon}</div>
                    {s.text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ── Message List ──────────────────────────────────────────── */
            <>
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="message-content">
                    <div
                      className={`message-bubble ${msg.isError ? 'error' : ''}`}
                      style={msg.isError ? { borderColor: 'rgba(239,68,68,0.3)' } : {}}
                    >
                      {msg.content}
                    </div>
                    {msg.source && msg.source !== 'unknown' && msg.references?.length > 0 && (
                      <div className={`message-source ${msg.source}`}>
                        Source: {[...new Set(msg.references.map(r => r.source_file))].join(', ')}
                      </div>
                    )}
                    {msg.source === 'unknown' && (
                      <div className="message-source unknown">
                        Source: General Knowledge
                      </div>
                    )}
                    {msg.references && msg.references.length > 0 && (
                      <div className="message-references">
                        <span className="references-label">📎 References:</span>
                        {msg.references.map((ref, j) => (
                          <span key={j} className="reference-pill" title={`${ref.source_file} — Section ${ref.section_number}`}>
                            {ref.policy_name} §{ref.section_number}: {ref.section_title}
                            <span className="reference-score">{Math.round(ref.relevance_score * 100)}%</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Typing indicator */}
              {isLoading && (
                <div className="message assistant">
                  <div className="message-avatar">🤖</div>
                  <div className="message-content">
                    <div className="message-bubble">
                      <div className="typing-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* ── Chat Input ──────────────────────────────────────────────── */}
        <div className="chat-input-area">
          <div className="chat-input-wrapper">
            <textarea
              ref={inputRef}
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask an HR question..."
              rows={1}
              disabled={isLoading}
            />
            <button
              className="btn-send"
              onClick={() => sendMessage()}
              disabled={isLoading || !input.trim()}
              title="Send message"
            >
              ➤
            </button>
          </div>
          <div className="chat-hint">
            Press Enter to send · Shift+Enter for new line
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
