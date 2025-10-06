/**
 * WebSocket client for Guardian Angel Dashboard
 */

class GuardianWebSocket {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.reconnectTimer = null;
        this.connected = false;
        this.listeners = {
            alert: [],
            systemStatus: [],
            cameraStatus: [],
            connection: []
        };

        this.connect();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        // Get auth credentials from sessionStorage (set during login)
        const username = sessionStorage.getItem('guardian_username') || 'admin';
        const password = sessionStorage.getItem('guardian_password') || 'change_me_in_production';

        // Add credentials as query parameters for WebSocket authentication
        const wsUrl = `${protocol}//${window.location.host}/ws?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`;

        console.log('Connecting to WebSocket...');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.notifyConnectionListeners(true);

            // Clear reconnect timer
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }

            // Start heartbeat
            this.startHeartbeat();
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.connected = false;
            this.notifyConnectionListeners(false);

            // Stop heartbeat
            this.stopHeartbeat();

            // Attempt reconnect
            this.scheduleReconnect();
        };
    }

    handleMessage(message) {
        const { type, data } = message;

        switch (type) {
            case 'alert':
                this.notifyListeners(this.listeners.alert, data);
                break;

            case 'system_status':
                this.notifyListeners(this.listeners.systemStatus, data);
                break;

            case 'camera_status':
                this.notifyListeners(this.listeners.cameraStatus, data);
                break;

            case 'connection':
                console.log('Connection confirmed:', message);
                break;

            case 'pong':
                // Heartbeat response
                break;

            default:
                console.log('Unknown message type:', type, message);
        }
    }

    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, cannot send message');
        }
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000); // Every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    scheduleReconnect() {
        if (!this.reconnectTimer) {
            this.reconnectTimer = setTimeout(() => {
                console.log('Attempting to reconnect...');
                this.connect();
            }, this.reconnectInterval);
        }
    }

    disconnect() {
        this.stopHeartbeat();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    // Listener registration methods
    onAlert(callback) {
        this.listeners.alert.push(callback);
    }

    onSystemStatus(callback) {
        this.listeners.systemStatus.push(callback);
    }

    onCameraStatus(callback) {
        this.listeners.cameraStatus.push(callback);
    }

    onStatusChange(callback) {
        this.listeners.connection.push(callback);
    }

    // Notify listeners
    notifyListeners(listeners, data) {
        listeners.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('Error in listener callback:', error);
            }
        });
    }

    notifyConnectionListeners(connected) {
        this.listeners.connection.forEach(callback => {
            try {
                callback(connected);
            } catch (error) {
                console.error('Error in connection listener:', error);
            }
        });
    }

    isConnected() {
        return this.connected;
    }
}

// Export for use in other scripts
window.GuardianWebSocket = GuardianWebSocket;
