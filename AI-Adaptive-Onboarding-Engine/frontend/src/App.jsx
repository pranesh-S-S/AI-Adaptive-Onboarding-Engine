import { useState, useRef, useEffect } from "react"
import axios from "axios"

// ─── STATUS CONFIG ────────────────────────────────────────────────────────────
const STATUS_CFG = {
  Completed:    { color: "#16a34a", bg: "#dcfce7", border: "#bbf7d0", dot: "#16a34a" },
  "In Progress":{ color: "#d97706", bg: "#fef3c7", border: "#fde68a", dot: "#d97706" },
  Start:        { color: "#4f46e5", bg: "#eef2ff", border: "#c7d2fe", dot: "#4f46e5" },
  Locked:       { color: "#9ca3af", bg: "#f3f4f6", border: "#e5e7eb", dot: "#d1d5db" },
}

// ─── ANIMATED SCORE RING ──────────────────────────────────────────────────────
function ScoreRing({ score, size = 130 }) {
  const stroke = 9, r = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  const [prog, setProg] = useState(0)
  useEffect(() => { const t = setTimeout(() => setProg(score), 250); return () => clearTimeout(t) }, [score])
  const color = score >= 75 ? "#16a34a" : score >= 50 ? "#d97706" : "#e11d48"
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#f1f5f9" strokeWidth={stroke} />
        <circle cx={size/2} cy={size/2} r={r} fill="none"
          stroke={color} strokeWidth={stroke}
          strokeDasharray={circ}
          strokeDashoffset={circ - (prog / 100) * circ}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1.4s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 26, fontWeight: 800, color: "#0f172a", fontFamily: "'Playfair Display', serif", lineHeight: 1 }}>{score}%</span>
        <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, letterSpacing: "0.06em", marginTop: 2 }}>SCORE</span>
      </div>
    </div>
  )
}

// ─── STAT BAR ─────────────────────────────────────────────────────────────────
function StatBar({ label, value, color = "#4f46e5" }) {
  const [w, setW] = useState(0)
  useEffect(() => { const t = setTimeout(() => setW(value), 400); return () => clearTimeout(t) }, [value])
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 14, color: "#475569", fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 14, color: "#0f172a", fontWeight: 700 }}>{value}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 99, background: "#f1f5f9", overflow: "hidden" }}>
        <div style={{ height: "100%", borderRadius: 99, background: color, width: `${w}%`, transition: "width 1.3s cubic-bezier(0.4,0,0.2,1)" }} />
      </div>
    </div>
  )
}

// ─── PILL ─────────────────────────────────────────────────────────────────────
function Pill({ label, variant = "gap" }) {
  const s = {
    gap:     { bg: "#fef2f2", border: "#fecaca", color: "#dc2626" },
    overlap: { bg: "#f0fdf4", border: "#bbf7d0", color: "#16a34a" },
    neutral: { bg: "#eef2ff", border: "#c7d2fe", color: "#4f46e5" },
  }[variant]
  return (
    <span style={{
      display: "inline-block", padding: "5px 13px", borderRadius: 99,
      fontSize: 13, fontWeight: 600, margin: "3px 3px 3px 0",
      background: s.bg, border: `1.5px solid ${s.border}`, color: s.color,
    }}>{label}</span>
  )
}

// ─── CARD ─────────────────────────────────────────────────────────────────────
function Card({ children, style = {}, accent = false }) {
  return (
    <div style={{
      background: "#fff",
      border: accent ? "1.5px solid #c7d2fe" : "1.5px solid #f1f5f9",
      borderRadius: 20,
      padding: 28,
      boxShadow: accent
        ? "0 4px 24px rgba(79,70,229,0.10)"
        : "0 2px 16px rgba(15,23,42,0.06)",
      ...style
    }}>{children}</div>
  )
}

function Label({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", color: "#94a3b8", marginBottom: 16 }}>{children}</div>
}

// ─── ROADMAP NODE (expandable) ────────────────────────────────────────────────
function RoadmapNode({ item, last }) {
  const [open, setOpen] = useState(false)
  const s = STATUS_CFG[item.status] || STATUS_CFG.Start
  const isLocked = item.status === "Locked"
  return (
    <div style={{ display: "flex", gap: 16, opacity: isLocked ? 0.5 : 1 }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 20, flexShrink: 0 }}>
        <div style={{ width: 14, height: 14, borderRadius: "50%", background: s.dot, marginTop: 16, flexShrink: 0, boxShadow: isLocked ? "none" : `0 0 0 4px ${s.bg}` }} />
        {!last && <div style={{ flex: 1, width: 2, background: "#f1f5f9", minHeight: 20 }} />}
      </div>
      <div style={{ flex: 1, marginBottom: last ? 0 : 12 }}>
        <div
          onClick={() => !isLocked && setOpen(o => !o)}
          style={{
            background: open ? "#fafbff" : "#fff",
            border: `1.5px solid ${open ? "#c7d2fe" : "#f1f5f9"}`,
            borderRadius: 14, padding: "16px 20px",
            cursor: isLocked ? "default" : "pointer",
            transition: "all 0.2s",
            boxShadow: open ? "0 2px 16px rgba(79,70,229,0.08)" : "0 1px 4px rgba(15,23,42,0.04)"
          }}
        >
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", fontFamily: "'Playfair Display', serif" }}>{item.skill}</span>
                <span style={{ fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 99, background: s.bg, color: s.color, border: `1px solid ${s.border}`, letterSpacing: "0.04em" }}>
                  {item.status}
                </span>
                {item.importance === "High" && !isLocked && (
                  <span style={{ fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 99, background: "#fef2f2", color: "#dc2626", border: "1px solid #fecaca" }}>HIGH</span>
                )}
              </div>
              <p style={{ fontSize: 13, color: "#64748b", margin: 0, lineHeight: 1.5 }}>{item.task}</p>
            </div>
            <div style={{ textAlign: "right", flexShrink: 0 }}>
              <div style={{ fontSize: 17, fontWeight: 800, color: "#4f46e5" }}>~{item.estimated_days}d</div>
              {!isLocked && <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>{open ? "▲" : "▼"}</div>}
            </div>
          </div>

          {open && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1.5px solid #f1f5f9" }}>
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>Exposure level</span>
                  <span style={{ fontSize: 12, color: "#4f46e5", fontWeight: 600 }}>{Math.round(item.progress * 100)}%</span>
                </div>
                <div style={{ height: 4, background: "#f1f5f9", borderRadius: 99 }}>
                  <div style={{ height: "100%", width: `${item.progress * 100}%`, background: "#4f46e5", borderRadius: 99 }} />
                </div>
              </div>
              {item.depends_on?.length > 0 && (
                <p style={{ fontSize: 12, color: "#64748b", margin: "0 0 8px" }}>
                  Requires: {item.depends_on.map((d, i) => <Pill key={i} label={d} variant="neutral" />)}
                </p>
              )}
              {item.resources?.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                  {item.resources.map((r, i) => (
                    <a key={i} href={r.link} target="_blank" rel="noopener noreferrer"
                      onClick={e => e.stopPropagation()}
                      style={{
                        display: "inline-flex", alignItems: "center", gap: 6,
                        padding: "6px 14px", borderRadius: 10, fontSize: 12, fontWeight: 600,
                        background: "#eef2ff", border: "1px solid #c7d2fe", color: "#4f46e5",
                        textDecoration: "none"
                      }}>
                      {r.type === "course" ? "🎓" : "📖"} {r.name}
                    </a>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── SLIDE DOTS NAV ───────────────────────────────────────────────────────────
function SlideDots({ total, current, onChange }) {
  return (
    <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <button key={i} onClick={() => onChange(i)} style={{
          width: current === i ? 28 : 8, height: 8, borderRadius: 99, border: "none",
          background: current === i ? "#4f46e5" : "#e2e8f0", cursor: "pointer",
          padding: 0, transition: "all 0.3s"
        }} />
      ))}
    </div>
  )
}

// ─── SVG ROBOT ────────────────────────────────────────────────────────────────
function RobotIllustration() {
  return (
    <svg viewBox="0 0 320 420" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ width: "100%", maxWidth: 340, filter: "drop-shadow(0 24px 48px rgba(79,70,229,0.18))" }}>
      <defs>
        <radialGradient id="bodyGrad" cx="50%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="100%" stopColor="#e0e7ff" />
        </radialGradient>
        <radialGradient id="headGrad" cx="50%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#312e81" />
          <stop offset="100%" stopColor="#1e1b4b" />
        </radialGradient>
        <radialGradient id="eyeGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#67e8f9" />
          <stop offset="100%" stopColor="#06b6d4" />
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <linearGradient id="armGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#e0e7ff"/>
          <stop offset="100%" stopColor="#c7d2fe"/>
        </linearGradient>
        <linearGradient id="shadowGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(79,70,229,0.15)"/>
          <stop offset="100%" stopColor="rgba(79,70,229,0)"/>
        </linearGradient>
      </defs>

      {/* ground shadow */}
      <ellipse cx="160" cy="408" rx="90" ry="12" fill="url(#shadowGrad)" />

      {/* antenna */}
      <line x1="160" y1="52" x2="160" y2="28" stroke="#c7d2fe" strokeWidth="4" strokeLinecap="round"/>
      <circle cx="160" cy="22" r="8" fill="#818cf8" filter="url(#glow)"/>
      <circle cx="160" cy="22" r="4" fill="#e0e7ff"/>

      {/* neck */}
      <rect x="148" y="148" width="24" height="18" rx="6" fill="#c7d2fe"/>

      {/* head */}
      <rect x="90" y="52" width="140" height="100" rx="28" fill="url(#headGrad)"/>
      {/* head highlight */}
      <rect x="100" y="58" width="120" height="40" rx="18" fill="rgba(255,255,255,0.07)"/>

      {/* visor panel */}
      <rect x="102" y="74" width="116" height="58" rx="16" fill="#0f172a"/>
      <rect x="105" y="77" width="110" height="52" rx="14" fill="#0d1526"/>

      {/* eyes */}
      <circle cx="135" cy="103" r="18" fill="url(#eyeGrad)" filter="url(#glow)"/>
      <circle cx="185" cy="103" r="18" fill="url(#eyeGrad)" filter="url(#glow)"/>
      <circle cx="135" cy="103" r="10" fill="#0891b2"/>
      <circle cx="185" cy="103" r="10" fill="#0891b2"/>
      <circle cx="138" cy="100" r="4" fill="#e0f2fe"/>
      <circle cx="188" cy="100" r="4" fill="#e0f2fe"/>
      {/* eye shine */}
      <circle cx="128" cy="96" r="3" fill="rgba(255,255,255,0.5)"/>
      <circle cx="178" cy="96" r="3" fill="rgba(255,255,255,0.5)"/>

      {/* smile */}
      <path d="M138 126 Q160 138 182 126" stroke="#67e8f9" strokeWidth="3" strokeLinecap="round" fill="none" filter="url(#glow)"/>

      {/* ear bolts */}
      <circle cx="90" cy="102" r="8" fill="#c7d2fe"/>
      <circle cx="90" cy="102" r="4" fill="#818cf8"/>
      <circle cx="230" cy="102" r="8" fill="#c7d2fe"/>
      <circle cx="230" cy="102" r="4" fill="#818cf8"/>

      {/* body */}
      <rect x="82" y="166" width="156" height="148" rx="32" fill="url(#bodyGrad)"/>
      {/* body highlight */}
      <rect x="92" y="172" width="136" height="60" rx="20" fill="rgba(255,255,255,0.7)"/>
      {/* body panel */}
      <rect x="106" y="200" width="108" height="80" rx="16" fill="#eef2ff" opacity="0.8"/>

      {/* chest light */}
      <circle cx="160" cy="228" r="14" fill="#4f46e5" opacity="0.15"/>
      <circle cx="160" cy="228" r="9" fill="#4f46e5" filter="url(#glow)"/>
      <circle cx="160" cy="228" r="5" fill="#a5b4fc"/>

      {/* chest buttons */}
      <circle cx="132" cy="256" r="5" fill="#c7d2fe"/>
      <circle cx="148" cy="256" r="5" fill="#a5b4fc"/>
      <circle cx="164" cy="256" r="5" fill="#818cf8"/>
      <circle cx="180" cy="256" r="5" fill="#6366f1"/>

      {/* left arm */}
      <rect x="34" y="172" width="44" height="120" rx="22" fill="url(#armGrad)"/>
      <rect x="38" y="176" width="36" height="60" rx="16" fill="rgba(255,255,255,0.5)"/>
      {/* left hand */}
      <ellipse cx="56" cy="302" rx="20" ry="16" fill="#c7d2fe"/>
      <rect x="42" y="292" width="10" height="22" rx="5" fill="#e0e7ff"/>
      <rect x="54" y="290" width="10" height="24" rx="5" fill="#e0e7ff"/>
      <rect x="66" y="294" width="10" height="20" rx="5" fill="#e0e7ff"/>

      {/* right arm */}
      <rect x="242" y="172" width="44" height="120" rx="22" fill="url(#armGrad)"/>
      <rect x="246" y="176" width="36" height="60" rx="16" fill="rgba(255,255,255,0.5)"/>
      {/* right hand */}
      <ellipse cx="264" cy="302" rx="20" ry="16" fill="#c7d2fe"/>
      <rect x="248" y="292" width="10" height="22" rx="5" fill="#e0e7ff"/>
      <rect x="260" y="290" width="10" height="24" rx="5" fill="#e0e7ff"/>
      <rect x="272" y="294" width="10" height="20" rx="5" fill="#e0e7ff"/>

      {/* left leg */}
      <rect x="108" y="308" width="44" height="80" rx="22" fill="#c7d2fe"/>
      <rect x="112" y="312" width="36" height="50" rx="16" fill="rgba(255,255,255,0.4)"/>
      {/* left foot */}
      <rect x="98" y="376" width="62" height="24" rx="12" fill="#a5b4fc"/>

      {/* right leg */}
      <rect x="168" y="308" width="44" height="80" rx="22" fill="#c7d2fe"/>
      <rect x="172" y="312" width="36" height="50" rx="16" fill="rgba(255,255,255,0.4)"/>
      {/* right foot */}
      <rect x="160" y="376" width="62" height="24" rx="12" fill="#a5b4fc"/>

      {/* floating sparkles */}
      <circle cx="52" cy="140" r="5" fill="#fbbf24" opacity="0.8"/>
      <circle cx="40" cy="160" r="3" fill="#f472b6" opacity="0.7"/>
      <circle cx="268" cy="148" r="4" fill="#34d399" opacity="0.8"/>
      <circle cx="280" cy="168" r="3" fill="#818cf8" opacity="0.7"/>
      <path d="M58 120 L62 128 L70 128 L64 134 L66 142 L58 137 L50 142 L52 134 L46 128 L54 128 Z" fill="#fbbf24" opacity="0.6"/>
    </svg>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// PAGE 1 — LANDING
// ════════════════════════════════════════════════════════════════════════════
function LandingPage({ onNext }) {
  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "60px 48px", position: "relative", overflow: "hidden" }}>
      {/* background blobs */}
      <div style={{ position: "fixed", top: -80, right: -80, width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, rgba(199,210,254,0.55) 0%, transparent 70%)", pointerEvents: "none" }} />
      <div style={{ position: "fixed", bottom: -60, left: -60, width: 380, height: 380, borderRadius: "50%", background: "radial-gradient(circle, rgba(253,230,138,0.35) 0%, transparent 70%)", pointerEvents: "none" }} />

      <div style={{
        width: "100%", maxWidth: 1100,
        display: "grid", gridTemplateColumns: "1fr auto",
        alignItems: "center", gap: 48, position: "relative", zIndex: 1
      }}>
        {/* LEFT — text */}
        <div style={{ maxWidth: 580 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8, padding: "7px 16px",
            borderRadius: 99, background: "#eef2ff", border: "1.5px solid #c7d2fe",
            color: "#4f46e5", fontSize: 12, fontWeight: 800, letterSpacing: "0.1em",
            textTransform: "uppercase", marginBottom: 32
          }}>🤖 AI-Powered Career Intelligence</div>

          <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: "clamp(44px, 6vw, 76px)", lineHeight: 1.08, fontWeight: 700, color: "#0f172a", margin: "0 0 6px" }}>
            Bridge the gap to
          </h1>
          <h1 style={{
            fontFamily: "'Playfair Display', serif", fontSize: "clamp(44px, 6vw, 76px)",
            lineHeight: 1.08, fontWeight: 700, margin: "0 0 28px",
            background: "linear-gradient(135deg, #4f46e5, #7c3aed)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
          }}>your dream role</h1>

          <p style={{ fontSize: 19, color: "#64748b", lineHeight: 1.75, margin: "0 0 44px", maxWidth: 460 }}>
            Upload your resume. Paste a job description. Get a precise skill gap analysis and a personalized learning roadmap — instantly.
          </p>

          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <button
              onClick={onNext}
              style={{
                padding: "17px 40px", borderRadius: 14,
                background: "linear-gradient(135deg, #4f46e5, #7c3aed)",
                border: "none", color: "#fff", fontSize: 17, fontWeight: 700,
                cursor: "pointer", letterSpacing: "0.01em",
                boxShadow: "0 8px 32px rgba(79,70,229,0.3)",
                transition: "transform 0.15s, box-shadow 0.15s",
                display: "flex", alignItems: "center", gap: 10
              }}
              onMouseOver={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 14px 40px rgba(79,70,229,0.4)" }}
              onMouseOut={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "0 8px 32px rgba(79,70,229,0.3)" }}
            >🚀 Analyze My Resume →</button>
          </div>

          <div style={{ display: "flex", gap: 28, marginTop: 52, flexWrap: "wrap" }}>
            {["ATS Scoring", "Skill Gap Analysis", "Adaptive Roadmap", "Learning Resources"].map(f => (
              <div key={f} style={{ display: "flex", alignItems: "center", gap: 7, color: "#94a3b8", fontSize: 14, fontWeight: 600 }}>
                <span style={{ color: "#a5b4fc" }}>◆</span> {f}
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT — robot */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          animation: "robotFloat 3.5s ease-in-out infinite"
        }}>
          <RobotIllustration />
        </div>
      </div>

      <style>{`
        @keyframes robotFloat {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-18px); }
        }
        @media (max-width: 700px) {
          .landing-grid { grid-template-columns: 1fr !important; text-align: center; }
          .robot-col { display: none !important; }
        }
      `}</style>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// PAGE 2 — UPLOAD
// ════════════════════════════════════════════════════════════════════════════
function DropZone({ label, accept, file, onFile, hint }) {
  const [drag, setDrag] = useState(false)
  const ref = useRef()
  return (
    <>
      <div
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onFile(f) }}
        onClick={() => ref.current.click()}
        style={{
          border: `2px dashed ${drag ? "#4f46e5" : file ? "#16a34a" : "#e2e8f0"}`,
          borderRadius: 14, padding: "40px 20px", textAlign: "center",
          cursor: "pointer", transition: "all 0.2s",
          background: drag ? "#eef2ff" : file ? "#f0fdf4" : "#fafbff"
        }}
      >
        <div style={{ fontSize: 40, marginBottom: 12 }}>{file ? "📄" : "☁️"}</div>
        <p style={{ fontSize: 15, fontWeight: 600, color: file ? "#16a34a" : "#64748b", margin: "0 0 4px" }}>
          {file ? file.name : label}
        </p>
        <p style={{ fontSize: 13, color: "#94a3b8", margin: 0 }}>{hint}</p>
      </div>
      <input ref={ref} type="file" style={{ display: "none" }} accept={accept} onChange={e => onFile(e.target.files[0])} />
    </>
  )
}

function UploadPage({ onSubmit, loading }) {
  const [resumeFile, setResumeFile] = useState(null)
  // JD mode: "text" | "file"
  const [jdMode, setJdMode]     = useState("text")
  const [jdText, setJdText]     = useState("")
  const [jdFile, setJdFile]     = useState(null)

  // derived: is the JD filled?
  const jdReady = jdMode === "text" ? jdText.trim().length > 0 : jdFile !== null
  const canSubmit = resumeFile && jdReady && !loading

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "48px 24px" }}>
      <div style={{ width: "100%", maxWidth: 900 }}>
        <div style={{ textAlign: "center", marginBottom: 44 }}>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 46, fontWeight: 700, color: "#0f172a", margin: "0 0 12px" }}>
            Let's get started
          </h2>
          <p style={{ fontSize: 18, color: "#64748b", margin: 0 }}>Upload your resume and provide the job description</p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20, marginBottom: 32 }}>

          {/* ── Resume card ── */}
          <Card>
            <Label>Your Resume</Label>
            <DropZone
              label="Drop your resume here"
              accept=".pdf,.docx"
              file={resumeFile}
              onFile={setResumeFile}
              hint="PDF or DOCX · Click to browse"
            />
          </Card>

          {/* ── JD card ── */}
          <Card>
            {/* mode toggle */}
            <div style={{ display: "flex", gap: 0, marginBottom: 18, background: "#f1f5f9", borderRadius: 10, padding: 3 }}>
              {[
                { id: "text", icon: "✏️", label: "Paste / Type" },
                { id: "file", icon: "📁", label: "Upload File"  },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setJdMode(tab.id)}
                  style={{
                    flex: 1, padding: "9px 0", borderRadius: 8, border: "none",
                    fontSize: 14, fontWeight: 700, cursor: "pointer", transition: "all 0.2s",
                    background: jdMode === tab.id ? "#fff" : "transparent",
                    color: jdMode === tab.id ? "#4f46e5" : "#94a3b8",
                    boxShadow: jdMode === tab.id ? "0 1px 6px rgba(79,70,229,0.12)" : "none",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 6
                  }}
                >
                  <span>{tab.icon}</span> {tab.label}
                </button>
              ))}
            </div>

            <Label>Job Description</Label>

            {jdMode === "text" ? (
              <>
                <textarea
                  value={jdText}
                  onChange={e => setJdText(e.target.value)}
                  placeholder={"Paste the job description here…\n\nor just type a role like 'ML Engineer'"}
                  style={{
                    width: "100%", minHeight: 168, padding: "14px", borderRadius: 12,
                    background: "#fafbff", border: "1.5px solid #e2e8f0",
                    color: "#0f172a", fontSize: 15, lineHeight: 1.6, resize: "vertical",
                    outline: "none", boxSizing: "border-box", fontFamily: "'Lato', sans-serif",
                    transition: "border-color 0.2s"
                  }}
                  onFocus={e => e.currentTarget.style.borderColor = "#a5b4fc"}
                  onBlur={e => e.currentTarget.style.borderColor = "#e2e8f0"}
                />
                <div style={{ marginTop: 12 }}>
                  {[
                    "Paste a full JD for best accuracy",
                    "Or just type a role title (e.g. 'ML Engineer')",
                    "Short titles are auto-expanded by AI",
                  ].map((t, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, marginBottom: 5 }}>
                      <span style={{ color: "#a5b4fc", fontWeight: 700, fontSize: 14 }}>✓</span>
                      <span style={{ fontSize: 13, color: "#94a3b8" }}>{t}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <>
                <DropZone
                  label="Drop your JD file here"
                  accept=".pdf,.docx,.txt"
                  file={jdFile}
                  onFile={setJdFile}
                  hint="PDF, DOCX or TXT · Click to browse"
                />
                <div style={{ marginTop: 12 }}>
                  {[
                    "Upload the job posting as PDF, DOCX or TXT",
                    "Text is extracted and analyzed automatically",
                    "Best for copy-pasted job listings saved as files",
                  ].map((t, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, marginBottom: 5 }}>
                      <span style={{ color: "#a5b4fc", fontWeight: 700, fontSize: 14 }}>✓</span>
                      <span style={{ fontSize: 13, color: "#94a3b8" }}>{t}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </Card>
        </div>

        <div style={{ textAlign: "center" }}>
          <button
            onClick={() => onSubmit(resumeFile, jdMode === "text" ? jdText : null, jdMode === "file" ? jdFile : null)}
            disabled={!canSubmit}
            style={{
              padding: "18px 52px", borderRadius: 14,
              background: canSubmit ? "linear-gradient(135deg, #4f46e5, #7c3aed)" : "#f1f5f9",
              border: "none",
              color: canSubmit ? "#fff" : "#94a3b8",
              fontSize: 18, fontWeight: 700,
              cursor: canSubmit ? "pointer" : "not-allowed",
              boxShadow: canSubmit ? "0 8px 32px rgba(79,70,229,0.25)" : "none",
              transition: "all 0.2s"
            }}
          >
            {loading ? "Analyzing…" : "Generate My Roadmap →"}
          </button>

          {loading && (
            <div style={{ marginTop: 24 }}>
              <div style={{ display: "inline-block", width: 220, height: 3, borderRadius: 99, background: "#f1f5f9", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg, transparent, #a5b4fc, transparent)", animation: "shimmer 1.5s infinite" }} />
              </div>
              <p style={{ color: "#94a3b8", fontSize: 15, marginTop: 12 }}>Running AI analysis — this takes 15–30s</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// PAGE 3 — DASHBOARD (4-slide)
// ════════════════════════════════════════════════════════════════════════════
const SLIDES = [
  { id: "overview", label: "Overview"    },
  { id: "skills",   label: "Skills"      },
  { id: "roadmap",  label: "Roadmap"     },
  { id: "weekly",   label: "Weekly Plan" },
]

function DashboardPage({ data, onReset }) {
  const [slide, setSlide] = useState(0)
  const score = data?.advanced_score
  const gap   = data?.gap_analysis
  const role  = data?.jd?.role_title || "Target Role"

  const readiness = score?.readiness || "Needs Improvement"
  const rStyle = {
    "Job Ready":              { bg: "#f0fdf4", color: "#16a34a", border: "#bbf7d0" },
    "Almost Ready":           { bg: "#fffbeb", color: "#d97706", border: "#fde68a" },
    "Needs Improvement":      { bg: "#fef2f2", color: "#dc2626", border: "#fecaca" },
    "Significant Gaps Found": { bg: "#fef2f2", color: "#dc2626", border: "#fecaca" },
  }[readiness] || { bg: "#fef2f2", color: "#dc2626", border: "#fecaca" }

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", paddingBottom: 80 }}>
      {/* sticky header */}
      <div style={{
        position: "sticky", top: 0, zIndex: 100,
        background: "rgba(255,255,255,0.95)", backdropFilter: "blur(12px)",
        borderBottom: "1.5px solid #f1f5f9",
        padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12
      }}>
        <div>
          <div style={{ fontSize: 12, color: "#94a3b8", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 2 }}>
            Role Analysis · {role}
          </div>
          <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 700, color: "#0f172a", margin: 0 }}>
            Your Career Roadmap
          </h1>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ padding: "8px 18px", borderRadius: 99, fontSize: 14, fontWeight: 700, background: rStyle.bg, color: rStyle.color, border: `1.5px solid ${rStyle.border}` }}>
            {readiness}
          </span>
          <button onClick={onReset} style={{ padding: "9px 18px", borderRadius: 10, background: "#f1f5f9", border: "none", color: "#64748b", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
            ← New Analysis
          </button>
        </div>
      </div>

      {/* slide tabs */}
      <div style={{ display: "flex", justifyContent: "center", gap: 8, padding: "28px 24px 0", flexWrap: "wrap" }}>
        {SLIDES.map((s, i) => (
          <button key={s.id} onClick={() => setSlide(i)} style={{
            padding: "11px 26px", borderRadius: 99, fontSize: 15, fontWeight: 700,
            border: slide === i ? "none" : "1.5px solid #e2e8f0",
            background: slide === i ? "linear-gradient(135deg,#4f46e5,#7c3aed)" : "#fff",
            color: slide === i ? "#fff" : "#64748b",
            cursor: "pointer", transition: "all 0.2s",
            boxShadow: slide === i ? "0 4px 16px rgba(79,70,229,0.25)" : "none"
          }}>{s.label}</button>
        ))}
      </div>

      {/* content */}
      <div style={{ maxWidth: 1000, margin: "28px auto 0", padding: "0 24px" }}>

        {/* ── SLIDE 0: Overview ── */}
        {slide === 0 && (
          <div style={{ display: "grid", gap: 20 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 20 }}>
              <Card style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 190 }}>
                <ScoreRing score={score?.final_score ?? 0} />
                <p style={{ fontSize: 13, color: "#94a3b8", marginTop: 14, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}>Final Score</p>
              </Card>
              <Card style={{ gridColumn: "span 2" }}>
                <Label>ATS Breakdown</Label>
                <StatBar label="Core Skills Match"  value={score?.required_score  ?? 0} color="#4f46e5" />
                <StatBar label="Preferred Skills"   value={score?.preferred_score ?? 0} color="#7c3aed" />
                <StatBar label="Skill Strength"     value={score?.strength_score  ?? 0} color="#16a34a" />
              </Card>
            </div>

            <Card>
              <Label>Skills at a Glance</Label>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", textAlign: "center" }}>
                {[
                  { val: gap?.overlapping_skills?.length ?? 0, label: "Matched", color: "#16a34a" },
                  { val: gap?.gap_skills?.length ?? 0,         label: "Missing",  color: "#dc2626" },
                  { val: gap?.level_gaps?.length ?? 0,         label: "Upskill",  color: "#d97706" },
                ].map((item, i, arr) => (
                  <div key={i} style={{ padding: "16px 0", borderRight: i < arr.length-1 ? "1.5px solid #f1f5f9" : "none" }}>
                    <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 46, fontWeight: 700, color: item.color, lineHeight: 1 }}>{item.val}</div>
                    <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 6, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em" }}>{item.label}</div>
                  </div>
                ))}
              </div>
            </Card>

            {data?.next_skill && data.next_skill.status !== "Locked" && (
              <Card accent>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
                  <div style={{ width: 54, height: 54, borderRadius: 14, background: "linear-gradient(135deg,#4f46e5,#7c3aed)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, flexShrink: 0, boxShadow: "0 4px 20px rgba(79,70,229,0.25)" }}>⚡</div>
                  <div style={{ flex: 1, minWidth: 180 }}>
                    <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.1em", color: "#94a3b8", textTransform: "uppercase", marginBottom: 4 }}>Start Here — Next Best Skill</div>
                    <h3 style={{ fontFamily: "'Playfair Display', serif", fontSize: 25, fontWeight: 700, color: "#0f172a", margin: "0 0 6px" }}>{data.next_skill.skill}</h3>
                    <p style={{ fontSize: 15, color: "#64748b", margin: 0 }}>{data.next_skill.task}</p>
                  </div>
                  <div style={{ textAlign: "right", flexShrink: 0 }}>
                    <div style={{ fontSize: 30, fontWeight: 800, color: "#4f46e5", fontFamily: "'Playfair Display', serif" }}>~{data.next_skill.estimated_days}</div>
                    <div style={{ fontSize: 13, color: "#94a3b8" }}>days</div>
                  </div>
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ── SLIDE 1: Skills ── */}
        {slide === 1 && (() => {
          const skillLevels  = data?.groq_skill_levels?.skills || {}
          const trace        = data?.gap_analysis?.reasoning_trace || {}
          const learningScore = data?.learning_score ?? 0

          const LEVEL_STYLE = {
            Advanced:     { bg: "#f0fdf4", color: "#16a34a", border: "#bbf7d0" },
            Intermediate: { bg: "#eff6ff", color: "#2563eb", border: "#bfdbfe" },
            Beginner:     { bg: "#fffbeb", color: "#d97706", border: "#fde68a" },
          }

          return (
            <div style={{ display: "grid", gap: 20 }}>

              {/* ── row 1: matched / gap pills + learning score ring ── */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 16, alignItems: "start" }}>
                <Card>
                  <Label>Your Matching Skills</Label>
                  <div style={{ lineHeight: 2.2 }}>
                    {gap?.overlapping_skills?.length
                      ? gap.overlapping_skills.map((s, i) => <Pill key={i} label={s} variant="overlap" />)
                      : <span style={{ fontSize: 15, color: "#94a3b8" }}>No overlapping skills detected</span>
                    }
                  </div>
                </Card>
                <Card>
                  <Label>Skills to Acquire</Label>
                  <div style={{ lineHeight: 2.2 }}>
                    {gap?.gap_skills?.length
                      ? gap.gap_skills.map((s, i) => <Pill key={i} label={s} variant="gap" />)
                      : <span style={{ fontSize: 15, color: "#16a34a", fontWeight: 600 }}>No gaps — great match! 🎉</span>
                    }
                  </div>
                </Card>

                {/* Learning Score ring */}
                <Card style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minWidth: 150, padding: "24px 20px" }}>
                  <Label>Learning Score</Label>
                  <ScoreRing score={Math.round(learningScore)} size={110} />
                  <p style={{ fontSize: 12, color: "#94a3b8", marginTop: 12, textAlign: "center", lineHeight: 1.4 }}>
                    Based on your current exposure to gap skills
                  </p>
                </Card>
              </div>

              {/* ── row 2: Groq Skill Level Diagnosis ── */}
              {Object.keys(skillLevels).length > 0 && (
                <Card>
                  <Label>Groq Skill Level Diagnosis</Label>
                  <p style={{ fontSize: 13, color: "#94a3b8", margin: "0 0 16px" }}>
                    AI-assessed proficiency levels extracted from your resume
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 10 }}>
                    {Object.entries(skillLevels).map(([skill, info], i) => {
                      const level = info?.level || "Beginner"
                      const ls = LEVEL_STYLE[level] || LEVEL_STYLE.Beginner
                      return (
                        <div key={i} style={{
                          display: "flex", flexDirection: "column", gap: 6,
                          padding: "14px 16px", borderRadius: 12,
                          background: "#fafbff", border: "1.5px solid #f1f5f9"
                        }}>
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                            <span style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>{skill}</span>
                            <span style={{
                              fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 99,
                              background: ls.bg, color: ls.color, border: `1px solid ${ls.border}`
                            }}>{level}</span>
                          </div>
                          {info?.reason && (
                            <p style={{ fontSize: 12, color: "#64748b", margin: 0, lineHeight: 1.5 }}>
                              {info.reason}
                            </p>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </Card>
              )}

              {/* ── row 3: Reasoning Trace ── */}
              {Object.keys(trace).length > 0 && (
                <Card>
                  <Label>Skill Matching Reasoning Trace</Label>
                  <p style={{ fontSize: 13, color: "#94a3b8", margin: "0 0 16px" }}>
                    Why each required skill was matched ✅ or flagged as missing ❌
                  </p>
                  <div style={{ display: "grid", gap: 8 }}>
                    {Object.entries(trace).map(([skill, t], i) => {
                      const covered = t.decision === "covered"
                      return (
                        <div key={i} style={{
                          display: "flex", alignItems: "center", gap: 14,
                          padding: "12px 16px", borderRadius: 12,
                          background: covered ? "#f0fdf4" : "#fef2f2",
                          border: `1.5px solid ${covered ? "#bbf7d0" : "#fecaca"}`
                        }}>
                          {/* icon */}
                          <span style={{ fontSize: 18, flexShrink: 0 }}>{covered ? "✅" : "❌"}</span>

                          {/* skill + best match */}
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                              <span style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>{skill}</span>
                              {t.best_match && t.best_match !== skill && (
                                <span style={{ fontSize: 12, color: "#64748b" }}>
                                  matched via <strong>{t.best_match}</strong>
                                </span>
                              )}
                            </div>
                            <p style={{ fontSize: 12, color: covered ? "#16a34a" : "#dc2626", margin: "2px 0 0", fontWeight: 600 }}>
                              {covered ? "Covered in your resume" : "Not found in your resume"}
                            </p>
                          </div>

                          {/* similarity score bar */}
                          <div style={{ textAlign: "right", flexShrink: 0, minWidth: 80 }}>
                            <div style={{ fontSize: 13, fontWeight: 700, color: covered ? "#16a34a" : "#dc2626" }}>
                              {Math.round((t.score || 0) * 100)}%
                            </div>
                            <div style={{ fontSize: 10, color: "#94a3b8", marginBottom: 4 }}>similarity</div>
                            <div style={{ width: 80, height: 4, background: "#e2e8f0", borderRadius: 99, overflow: "hidden" }}>
                              <div style={{
                                height: "100%", borderRadius: 99,
                                width: `${(t.score || 0) * 100}%`,
                                background: covered ? "#16a34a" : "#dc2626"
                              }} />
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </Card>
              )}

              {/* ── row 4: upskill level gaps ── */}
              {gap?.level_gaps?.length > 0 && (
                <Card>
                  <Label>Skills Needing Upskilling</Label>
                  <div style={{ display: "grid", gap: 10 }}>
                    {gap.level_gaps.map((lg, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 18px", borderRadius: 12, background: "#fffbeb", border: "1.5px solid #fde68a" }}>
                        <span style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>{lg.skill}</span>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <span style={{ fontSize: 13, padding: "4px 12px", borderRadius: 99, background: "#fef3c7", color: "#92400e", fontWeight: 600 }}>{lg.current_level}</span>
                          <span style={{ color: "#d97706", fontWeight: 700, fontSize: 16 }}>→</span>
                          <span style={{ fontSize: 13, padding: "4px 12px", borderRadius: 99, background: "#dcfce7", color: "#14532d", fontWeight: 600 }}>{lg.required_level}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

            </div>
          )
        })()}

        {/* ── SLIDE 2: Roadmap ── */}
        {slide === 2 && (
          <Card>
            <Label>Adaptive Learning Roadmap</Label>
            <p style={{ fontSize: 15, color: "#94a3b8", margin: "0 0 24px" }}>Click any skill to expand resources and details</p>
            <div>
              {data?.roadmap?.map((item, i) => (
                <RoadmapNode key={i} item={item} last={i === data.roadmap.length - 1} />
              ))}
            </div>
          </Card>
        )}

        {/* ── SLIDE 3: Weekly Plan ── */}
        {slide === 3 && (
          <div style={{ display: "grid", gap: 16 }}>
            {data?.weekly_roadmap?.length > 0
              ? data.weekly_roadmap.map((week, wi) => {
                  const days = week.reduce((a, b) => a + b.days, 0)
                  return (
                    <Card key={wi}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                        <h3 style={{ fontFamily: "'Playfair Display', serif", fontSize: 24, fontWeight: 700, color: "#0f172a", margin: 0 }}>Week {wi + 1}</h3>
                        <span style={{ fontSize: 14, color: "#94a3b8", fontWeight: 600, padding: "5px 14px", borderRadius: 99, background: "#f1f5f9" }}>{days} working days</span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10 }}>
                        {week.map((item, i) => (
                          <div key={i} style={{ padding: "14px 16px", borderRadius: 12, background: "#fafbff", border: "1.5px solid #e0e7ff" }}>
                            <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", marginBottom: 4 }}>{item.skill}</div>
                            <div style={{ fontSize: 14, color: "#94a3b8" }}>~{item.days} days</div>
                            <div style={{ marginTop: 8 }}>
                              <span style={{
                                fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 99,
                                background: item.importance === "High" ? "#fef2f2" : "#f0fdf4",
                                color: item.importance === "High" ? "#dc2626" : "#16a34a",
                                border: `1px solid ${item.importance === "High" ? "#fecaca" : "#bbf7d0"}`
                              }}>{item.importance}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </Card>
                  )
                })
              : <Card><p style={{ fontSize: 16, color: "#94a3b8", textAlign: "center", padding: "48px 0" }}>No weekly plan available.</p></Card>
            }
          </div>
        )}

        {/* prev / next + dots */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 36 }}>
          <button onClick={() => setSlide(s => Math.max(s - 1, 0))} disabled={slide === 0} style={{
            padding: "12px 28px", borderRadius: 12, fontSize: 15, fontWeight: 600,
            border: "1.5px solid #e2e8f0", background: slide === 0 ? "#f8fafc" : "#fff",
            color: slide === 0 ? "#cbd5e1" : "#475569", cursor: slide === 0 ? "default" : "pointer"
          }}>← Previous</button>

          <SlideDots total={SLIDES.length} current={slide} onChange={setSlide} />

          <button onClick={() => setSlide(s => Math.min(s + 1, SLIDES.length - 1))} disabled={slide === SLIDES.length - 1} style={{
            padding: "12px 28px", borderRadius: 12, fontSize: 15, fontWeight: 700,
            border: "none",
            background: slide === SLIDES.length - 1 ? "#f1f5f9" : "linear-gradient(135deg,#4f46e5,#7c3aed)",
            color: slide === SLIDES.length - 1 ? "#94a3b8" : "#fff",
            cursor: slide === SLIDES.length - 1 ? "default" : "pointer",
            boxShadow: slide === SLIDES.length - 1 ? "none" : "0 4px 16px rgba(79,70,229,0.25)"
          }}>Next →</button>
        </div>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// ROOT
// ════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage]       = useState("landing")
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)

  // resumeFile — the resume File object
  // jdText     — raw string (or null if using file mode)
  // jdFile     — File object (or null if using text mode)
  const handleSubmit = async (resumeFile, jdText, jdFile) => {
    if (!resumeFile) return
    if (!jdText?.trim() && !jdFile) return

    const formData = new FormData()
    formData.append("resume", resumeFile)

    if (jdFile) {
      // file mode — backend saves it and passes the path to process_job_description()
      formData.append("jd_file", jdFile)
    } else {
      // text mode — backend passes the raw string to process_job_description()
      formData.append("job_description", jdText.trim())
    }

    try {
      setLoading(true)
      const res = await axios.post("/analyze", formData)
      setData(res.data)
      setPage("dashboard")
    } catch {
      alert("Backend error — make sure the API server is running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@400;500;600;700;800&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { min-height: 100%; font-family: 'Lato', sans-serif; background: #f8fafc; color: #0f172a; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f1f5f9; }
        ::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 99px; }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
        textarea::placeholder { color: #94a3b8; }
      `}</style>

      {page === "landing"   && <LandingPage  onNext={() => setPage("upload")} />}
      {page === "upload"    && <UploadPage   onSubmit={handleSubmit} loading={loading} />}
      {page === "dashboard" && <DashboardPage data={data} onReset={() => { setPage("landing"); setData(null) }} />}
    </>
  )
}
