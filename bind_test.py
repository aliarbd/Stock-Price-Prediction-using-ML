import socket
for addr in [('127.0.0.1', 8001), ('::1', 8001)]:
    af = socket.AF_INET if ':' not in addr[0] else socket.AF_INET6
    s = socket.socket(af, socket.SOCK_STREAM)
    try:
        if af == socket.AF_INET6:
            s.bind((addr[0], addr[1], 0, 0))
        else:
            s.bind(addr)
        print('bind ok', addr)
    except Exception as e:
        print('bind failed', addr, type(e).__name__, e)
    finally:
        s.close()
