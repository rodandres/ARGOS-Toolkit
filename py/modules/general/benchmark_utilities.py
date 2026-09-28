import os
import time
import psutil
import threading


def benchmark(func, *args, **kwargs):
    process = psutil.Process(os.getpid())

    # Estado inicial
    ram_inicial = process.memory_info().rss
    cpu_inicial = process.cpu_times()

    ram_maxima = ram_inicial
    ejecutando = True

    def monitor_ram():
        nonlocal ram_maxima

        while ejecutando:
            ram_actual = process.memory_info().rss
            ram_maxima = max(ram_maxima, ram_actual)
            time.sleep(0.001)  # cada 1 ms

    monitor = threading.Thread(target=monitor_ram)

    # Ejecutar
    inicio = time.perf_counter()
    monitor.start()

    resultado = func(*args, **kwargs)

    fin = time.perf_counter()

    # Detener monitor
    ejecutando = False
    monitor.join()

    # Estado final
    ram_final = process.memory_info().rss
    cpu_final = process.cpu_times()

    # Cálculos
    tiempo = fin - inicio

    cpu = (
        (cpu_final.user - cpu_inicial.user) +
        (cpu_final.system - cpu_inicial.system)
    )

    stats = {
            "resultado": resultado,
            "tiempo_s": tiempo,
            "tiempo_ms": tiempo * 1000,
            "cpu_s": cpu,
            "cpu_ms": cpu * 1000,
            "ram_inicial_mb": ram_inicial / 1024**2,
            "ram_final_mb": ram_final / 1024**2,
            "ram_maxima_mb": ram_maxima / 1024**2,
            "ram_incremento_mb": (ram_final - ram_inicial) / 1024**2,
        }

    print(f"Tiempo:       {stats['tiempo_ms']:.2f} ms")
    print(f"CPU:          {stats['cpu_ms']:.2f} ms")
    print(f"RAM inicial:  {stats['ram_inicial_mb']:.2f} MB")
    print(f"RAM máxima:   {stats['ram_maxima_mb']:.2f} MB")
    print(f"RAM final:    {stats['ram_final_mb']:.2f} MB")


    
