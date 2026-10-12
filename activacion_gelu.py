import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import math

def gelu_matematica(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

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
    x = np.linspace(-4, 4, 100)
    
    y_math = gelu_matematica(x)
    print("--- Evaluación Matemática (Aproximación NumPy) ---")
    print(f"Entrada negativa pequeña (-1): {gelu_matematica(-1.0):.4f}")
    print(f"Entrada 0: {gelu_matematica(0.0):.4f}")
    print(f"Entrada positiva pequeña (1): {gelu_matematica(1.0):.4f}\n")

    modelo = keras.Sequential([
        keras.layers.Dense(1, activation='gelu', input_shape=(1,))
    ])
    modelo.set_weights([np.array([[1.0]]), np.array([0.0])])
    y_keras = modelo.predict(x)

    modelo.summary()
    graficar_neurona("GELU", "gelu_neurona.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_math, label="GELU (Aproximación NumPy)", color='teal', linewidth=2)
    plt.plot(x, y_keras.flatten(), label="GELU (Keras Exacto)", color='magenta', linestyle='-.')
    plt.title("Función de Activación: GELU")
    plt.xlabel("Entrada (x)")
    plt.ylabel("Salida (y)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    plt.savefig("gelu_plot.png", dpi=300)
    print("Gráfico guardado como 'gelu_plot.png'")

if __name__ == "__main__":
    main()