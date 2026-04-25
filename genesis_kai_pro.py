"""
GÉNESIS-KAI Pro - Explorador de diseños para actuadores Maglev
Autor: Abraham Hernández Dorantes
Descripción: Evalúa múltiples alternativas de ingeniería (Rth, I_peak, N, material)
con validación dual continua (T<T_critica y ∇B≥∇B_min) y análisis del transitorio.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================================
# NÚCLEO FÍSICO (igual que antes, pero ligeramente parametrizado)
# ============================================================================

class ThermalElectroSolver:
    """Solver térmico-magnético con parámetros configurables."""
    def __init__(self, R0=0.02, masa_cu=15.0, Cp_cu=385.0, alpha_cu=0.00393,
                 T_amb=25.0, Rth=0.8, N=450, gap=0.1, k_h=0.002, f_pwm=500.0,
                 T_critica=110.0, gradB_minimo=55.0):
        self.R0 = R0
        self.masa_cu = masa_cu
        self.Cp_cu = Cp_cu
        self.alpha_cu = alpha_cu
        self.T_amb = T_amb
        self.Rth = Rth
        self.N = N
        self.gap = gap
        self.k_h = k_h
        self.f_pwm = f_pwm
        self.T_critica = T_critica
        self.gradB_minimo = gradB_minimo

    def calculate_force_gradient(self, I_inst):
        mu0 = 4e-7 * np.pi
        B = (mu0 * self.N * I_inst) / self.gap
        gradB = B / self.gap
        return B, gradB

    def simulate_step(self, dt, I_actual, T_prev, Rth_current):
        R_actual = self.R0 * (1 + self.alpha_cu * (T_prev - self.T_amb))
        P_joule = (I_actual ** 2) * R_actual
        B_inst, gradB_inst = self.calculate_force_gradient(I_actual)
        P_hyst = self.k_h * self.f_pwm * (B_inst ** 2) if I_actual > 0 else 0
        P_total = P_joule + P_hyst
        C_th = self.masa_cu * self.Cp_cu
        dT = ((P_total - (T_prev - self.T_amb) / Rth_current) / C_th) * dt
        T_new = T_prev + dT
        return T_new, gradB_inst


# ============================================================================
# SIMULADOR DE MISIÓN CON VALIDACIÓN DUAL Y ANÁLISIS DE TRANSITORIO
# ============================================================================

def run_mission(solver, I_peak, duty_initial=0.4, duration=5.0, dt=0.0001, 
                enable_control=True, Rth_degradation=False):
    """
    Ejecuta una misión y devuelve:
        - éxito (bool): True si nunca se violaron T_critica ni gradB_minimo.
        - métricas: diccionario con análisis transitorio.
        - historial completo (opcional, para graficar).
    """
    t_array = np.arange(0, duration, dt)
    periodo = 1.0 / solver.f_pwm

    T_actual = solver.T_amb
    duty_actual = duty_initial
    Rth_actual = solver.Rth
    if Rth_degradation:
        Rth_deg = solver.Rth
    else:
        Rth_deg = solver.Rth  # sin degradación adicional

    exito_termico = True
    exito_magnetico = True
    tiempo_critico = None   # primer instante donde T > T_critica
    t_max_temp = 0.0
    temp_max = T_actual
    gradB_min = float('inf')
    transitorio_T_max = T_actual   # máximo en los primeros 2 segundos
    transitorio_end_idx = int(2.0 / dt)

    history = {'t': [], 'T': [], 'duty': [], 'gradB_peak': []}

    for idx, t in enumerate(t_array):
        ancho_pulso = periodo * duty_actual
        I_actual = I_peak if (t % periodo) < ancho_pulso else 0.0

        T_actual, gradB_inst = solver.simulate_step(dt, I_actual, T_actual, Rth_deg)
        _, gradB_peak = solver.calculate_force_gradient(I_peak)

        # Control dinámico (opcional)
        if enable_control:
            if T_actual > 105.0:
                duty_actual = max(0.15, duty_actual - 0.0001)
            elif T_actual < 90.0:
                duty_actual = min(0.60, duty_actual + 0.00005)

        if Rth_degradation and T_actual > solver.T_critica:
            Rth_deg += 0.000002

        # Validación dual continua
        if T_actual > solver.T_critica:
            exito_termico = False
            if tiempo_critico is None:
                tiempo_critico = t
        if gradB_peak < solver.gradB_minimo:
            exito_magnetico = False

        # Métricas de transitorio
        if T_actual > temp_max:
            temp_max = T_actual
            t_max_temp = t
        if gradB_inst < gradB_min:
            gradB_min = gradB_inst
        if idx <= transitorio_end_idx and T_actual > transitorio_T_max:
            transitorio_T_max = T_actual

        # Registrar historial (solo cada 1 ms para no saturar)
        if int(t / dt) % 10 == 0:
            history['t'].append(t)
            history['T'].append(T_actual)
            history['duty'].append(duty_actual)
            history['gradB_peak'].append(gradB_peak)

    exito = exito_termico and exito_magnetico

    metricas = {
        'exito_termico': exito_termico,
        'exito_magnetico': exito_magnetico,
        'exito': exito,
        'temp_max': temp_max,
        't_max_temp': t_max_temp,
        'tiempo_critico': tiempo_critico,
        'gradB_min': gradB_min,
        'transitorio_T_max': transitorio_T_max,
        'duty_final': duty_actual,
        'Rth_final': Rth_deg,
    }
    return metricas, history


# ============================================================================
# EXPLORADOR DE ALTERNATIVAS DE INGENIERÍA
# ============================================================================

def explorar_disenos(param_grid, fixed_params=None, duration=5.0):
    """
    Itera sobre todas las combinaciones de parámetros en param_grid y ejecuta
    la misión, devolviendo un DataFrame (lista de dicts) con los resultados.
    """
    if fixed_params is None:
        fixed_params = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    resultados = []

    for combination in product(*values):
        params = dict(zip(keys, combination))
        params.update(fixed_params)

        # Crear solver con los parámetros actuales
        solver = ThermalElectroSolver(
            R0=params.get('R0', 0.02),
            masa_cu=params.get('masa_cu', 15.0),
            Rth=params.get('Rth', 0.8),
            N=params.get('N', 450),
            T_critica=params.get('T_critica', 110.0),
            gradB_minimo=params.get('gradB_minimo', 55.0),
            k_h=params.get('k_h', 0.002),
            f_pwm=params.get('f_pwm', 500.0)
        )

        I_peak = params.get('I_peak', 450.0)
        duty_initial = params.get('duty_initial', 0.4)
        enable_control = params.get('enable_control', True)
        rth_degradation = params.get('rth_degradation', False)

        metricas, _ = run_mission(
            solver, I_peak, duty_initial, duration,
            enable_control=enable_control,
            Rth_degradation=rth_degradation
        )

        resultado = {
            'Rth': params['Rth'],
            'I_peak': I_peak,
            'N': params['N'],
            'T_critica': params['T_critica'],
            'exito_termico': metricas['exito_termico'],
            'exito_magnetico': metricas['exito_magnetico'],
            'exito': metricas['exito'],
            'temp_max': metricas['temp_max'],
            'transitorio_T_max': metricas['transitorio_T_max'],
            'tiempo_critico': metricas['tiempo_critico'],
            'gradB_min': metricas['gradB_min'],
            'duty_final': metricas['duty_final'],
        }
        resultados.append(resultado)
        print(f"Probado: Rth={params['Rth']} | I_peak={I_peak} | N={params['N']} | "
              f"T_crit={params['T_critica']} → Éxito: {metricas['exito']} "
              f"(T_max={metricas['temp_max']:.1f}°C, ∇B_min={metricas['gradB_min']:.1f} T/m)")

    return resultados


def mostrar_reporte(resultados):
    """Imprime una tabla formateada y un resumen de los diseños exitosos."""
    print("\n" + "="*80)
    print("RESUMEN DE DISEÑOS EVALUADOS")
    print("="*80)
    print(f"{'Rth':>6} {'I_peak':>8} {'N':>6} {'T_crit':>8} {'Éxito':>6} {'T_max(°C)':>10} {'∇B_min(T/m)':>12} {'T_trans(°C)':>10}")
    print("-"*80)
    for r in resultados:
        print(f"{r['Rth']:6.2f} {r['I_peak']:8.1f} {r['N']:6d} {r['T_critica']:8.1f} "
              f"{str(r['exito']):>6} {r['temp_max']:10.1f} {r['gradB_min']:12.1f} {r['transitorio_T_max']:10.1f}")

    exitosos = [r for r in resultados if r['exito']]
    print("\n" + "="*80)
    print(f"🔧 DISEÑOS QUE CUMPLEN EL CRITERIO DUAL (T<110°C Y ∇B≥55 T/m): {len(exitosos)} de {len(resultados)}")
    if exitosos:
        print("Parámetros recomendados:")
        for e in exitosos:
            print(f"  • Rth={e['Rth']:.2f} °C/W, I_peak={e['I_peak']:.0f} A, N={e['N']}, T_crit={e['T_critica']:.0f}°C → T_max={e['temp_max']:.1f}°C, ∇B_min={e['gradB_min']:.1f} T/m")
    else:
        print("⚠️ Ningún diseño cumple. Se requiere enfriamiento más agresivo (Rth más bajo) o mayor corriente crítica.")
    print("="*80)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Definir el espacio de búsqueda de alternativas
    param_grid = {
        'Rth': [0.4, 0.6, 0.8, 1.0],          # Diferentes capacidades de enfriamiento (mejor a peor)
        'I_peak': [350, 450, 550],             # Corriente pico (menos corriente = menos calor pero menos fuerza)
        'N': [400, 450, 500],                  # Espiras (aumentar espiras reduce I_peak necesario)
        'T_critica': [110, 130, 150],          # Materiales de aislante: estándar, mejor, avanzado
    }

    fixed = {
        'R0': 0.02,
        'masa_cu': 15.0,
        'gap': 0.1,
        'k_h': 0.002,
        'f_pwm': 500.0,
        'gradB_minimo': 55.0,
        'enable_control': True,
        'rth_degradation': False,   # Para análisis de diseño, no activamos degradación (evaluamos caso nominal)
        'duration': 5.0,
    }

    print("🚄 GÉNESIS-KAI Pro - Explorando alternativas de ingeniería para JR Central...")
    resultados = explorar_disenos(param_grid, fixed_params=fixed, duration=5.0)
    mostrar_reporte(resultados)

    # Opcional: graficar un diseño exitoso de ejemplo
    # Elegir el primer diseño exitoso para graficar (si existe)
    exitosos = [r for r in resultados if r['exito']]
    if exitosos:
        mejor = exitosos[0]
        print(f"\n📊 Graficando comportamiento del diseño exitoso: Rth={mejor['Rth']} °C/W, I_peak={mejor['I_peak']} A, N={mejor['N']}, T_crit={mejor['T_critica']}°C")
        # Re-ejecutar misión con ese diseño para obtener historial
        solver = ThermalElectroSolver(Rth=mejor['Rth'], N=mejor['N'], T_critica=mejor['T_critica'])
        _, history = run_mission(solver, I_peak=mejor['I_peak'], duration=5.0)
        # Graficar (similar a la versión original)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(history['t'], history['T'], color='red')
        axs[0].axhline(mejor['T_critica'], color='black', linestyle='--', label=f'T_crítica ({mejor["T_critica"]}°C)')
        axs[0].set_ylabel('Temperatura pico (°C)')
        axs[0].legend()
        axs[1].plot(history['t'], history['gradB_peak'], color='green')
        axs[1].axhline(55.0, color='orange', linestyle=':', label='∇B mínimo (55 T/m)')
        axs[1].set_ylabel('Gradiente magnético (T/m)')
        axs[1].set_xlabel('Tiempo (s)')
        axs[1].legend()
        plt.suptitle(f'Diseño exitoso: Rth={mejor["Rth"]} °C/W, I_peak={mejor["I_peak"]} A, N={mejor["N"]}')
        plt.tight_layout()
        plt.show()
