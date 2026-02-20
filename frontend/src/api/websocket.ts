import { useEffect, useRef, useCallback } from 'react'

type FrameHandler = (data: Record<string, unknown>) => void

export function useRadarWebSocket(onFrame: FrameHandler) {
    const wsRef = useRef<WebSocket | null>(null)
    const handlerRef = useRef(onFrame)
    handlerRef.current = onFrame

    const connect = useCallback(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
        const ws = new WebSocket(`${protocol}://${window.location.host}/ws/stream`)
        wsRef.current = ws

        ws.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data)
                if (data.type !== 'ping') handlerRef.current(data)
            } catch (_) { }
        }

        ws.onclose = () => {
            // Auto-reconnect after 2 seconds
            setTimeout(connect, 2000)
        }

        ws.onerror = () => ws.close()
    }, [])

    useEffect(() => {
        connect()
        return () => {
            wsRef.current?.close()
        }
    }, [connect])
}
