import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

def linear_matematica(x):
    return x

def graficar_neurona(nombre_activacion, archivo_salida, w=1.0, b=0.0):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    ax.add_patch(plt.Circle((1.5, 2), 0.5, color='skyblue', ec='black', zorder=3))
    ax.text(1.5, 2, 'x', ha='center', va='center', fontsize=14, zorder=4)

    ax.add_patch(plt.Circle((5, 2), 0.8, color='lightgreen', ec='black', zorder=3))
    ax.text(5, 2, f'Σ\n{nombre_activacion}', ha='center', va='center', fontsize=10, zorder=4)

    ax.add_patch(plt.Circle((8.5, 2), 0.5, color='salmon', ec='black', zorder=3))
    ax.text(8.5, 2, 'y', ha='center', va='center', fontsize=14, zorder=4)

    ax.annotate('', xy=(4.2, 2), xytext=(2, 2), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(3, 2.3, f'w={w}', ha='center', fontsize=9)
    ax.annotate('', xy=(8, 2), xytext=(5.8, 2), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(6.9, 2.3, f'b={b}', ha='center', fontsize=9)

    ax.set_title(f'Neurona con activación: {nombre_activacion}')
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Diagrama de neurona guardado como '{archivo_salida}'")

def main():
    x = np.linspace(-10, 10, 100)
    
    y_math = linear_matematica(x)
    print("--- Evaluación Matemática (NumPy) ---")
    print(f"Entradas: {x[0:3]}")
    print(f"Salidas:  {y_math[0:3]} (Identidad)\n")

    modelo = keras.Sequential([
        keras.layers.Dense(1, activation='linear', input_shape=(1,))
    ])
    modelo.set_weights([np.array([[1.0]]), np.array([0.0])])
    y_keras = modelo.predict(x)

    modelo.summary()
    graficar_neurona("Lineal", "linear_neurona.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_math, label="Lineal / Identidad", color='brown', linewidth=2)
    plt.title("Función de Activación: Lineal")
    plt.xlabel("Entrada (x)")
    plt.ylabel("Salida (y)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    plt.savefig("linear_plot.png", dpi=300)
    print("Gráfico guardado como 'linear_plot.png'")

if __name__ == "__main__":
    main()