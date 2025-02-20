import nmap

def escanear_ip(ip_objetivo, puertos="1-1024"):
    # Crear el escáner nmap
    escaner = nmap.PortScanner()

    # Ejecutar el escaneo en la IP objetivo y en el rango de puertos especificado
    print(f"Escaneando la IP {ip_objetivo} en los puertos {puertos}...")
    escaner.scan(ip_objetivo, puertos)

    # Verificar si la IP es accesible
    if ip_objetivo in escaner.all_hosts():
        # Imprimir los resultados del escaneo
        print(f"Resultados del escaneo para {ip_objetivo}:")
        for puerto in escaner[ip_objetivo]['tcp']:
            estado = escaner[ip_objetivo]['tcp'][puerto]['state']
            servicio = escaner[ip_objetivo]['tcp'][puerto]['name']
            print(f"Puerto: {puerto}\t Estado: {estado}\t Servicio: {servicio}")
    else:
        print(f"No se pudo acceder a la IP {ip_objetivo}. Puede estar fuera de línea o bloqueando el escaneo.")

# Definir la IP y el rango de puertos que deseas escanear
ip = "192.168.1.139"  # Cambia esta IP por la IP de tu red local que quieres escanear
puertos = "1-1024"  # Rango de puertos que quieres escanear

# Ejecutar la función de escaneo
escanear_ip(ip, puertos)
