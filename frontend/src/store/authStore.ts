import { create } from 'zustand'

interface AuthState {
    token: string | null
    username: string | null
    role: string | null
    login: (token: string, username: string, role: string) => void
    logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
    token: localStorage.getItem('aegis_token'),
    username: localStorage.getItem('aegis_user'),
    role: localStorage.getItem('aegis_role'),
    login: (token, username, role) => {
        localStorage.setItem('aegis_token', token)
        localStorage.setItem('aegis_user', username)
        localStorage.setItem('aegis_role', role)
        set({ token, username, role })
    },
    logout: () => {
        localStorage.removeItem('aegis_token')
        localStorage.removeItem('aegis_user')
        localStorage.removeItem('aegis_role')
        set({ token: null, username: null, role: null })
    },
}))
