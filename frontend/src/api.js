// Empty base uses the current host. Vite/Nginx proxies /api to Flask,
// so phones on the same network never call their own localhost.
const BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, { credentials: 'include', ...options })
  } catch (error) {
    throw new Error('เชื่อมต่อ AI Backend ไม่ได้ กรุณาตรวจว่า Flask เปิดอยู่ แล้วรีเฟรชหน้าเว็บ')
  }
  const body = await response.json().catch(() => ({ error: 'เซิร์ฟเวอร์ส่งข้อมูลไม่ถูกต้อง' }))
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`)
  return body
}

export const api = {
  health: () => request('/api/v1/health'),
  pipeline: (file) => {
    const data = new FormData(); data.append('image', file)
    return request('/api/v1/predict/pipeline', { method: 'POST', body: data })
  },
  identify: (file) => {
    const data = new FormData(); data.append('image', file)
    return request('/api/v1/predict/identify', { method: 'POST', body: data })
  },
  ripeness: (file) => {
    const data = new FormData(); data.append('image', file)
    return request('/api/v1/predict/ripeness', { method: 'POST', body: data })
  },
  replaceModel: (slot, file, displayName = '') => {
    const data = new FormData(); data.append('model', file); data.append('slot', slot); data.append('display_name', displayName)
    return request('/api/v1/admin/models/replace', { method: 'POST', body: data })
  },
  adminSession: () => request('/api/v1/admin/session'),
  adminModels: () => request('/api/v1/admin/models'),
  adminLogin: (username, password) => request('/api/v1/admin/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password })
  }),
  adminLogout: () => request('/api/v1/admin/logout', { method: 'POST' })
}
