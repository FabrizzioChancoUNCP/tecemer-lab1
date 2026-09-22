import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

def softmax_matematica(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

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
    logits = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 0.1],
        [-1.0, 0.0, 1.0],
        [5.0, 5.0, 5.0],
        [0.0, -2.0, -4.0]
    ])
    
    probabilidades_math = softmax_matematica(logits)
    print("--- Evaluación Matemática (NumPy) ---")
    for i, (log, prob) in enumerate(zip(logits, probabilidades_math)):
        print(f"Logits {i}: {log} -> Probabilidades: {prob.round(3)} (Suma: {np.sum(prob):.1f})")
    print()

    modelo = keras.Sequential([
        keras.layers.Dense(3, activation='softmax', input_shape=(3,))
    ])
    modelo.set_weights([np.eye(3), np.zeros(3)])
    probabilidades_keras = modelo.predict(logits)

    modelo.summary()
    graficar_neurona("Softmax", "softmax_neurona.png")

    clases = ['Clase A', 'Clase B', 'Clase C']
    ejemplo_idx = 0 # Visualizaremos el primer ejemplo
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clases, probabilidades_math[ejemplo_idx], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f"Distribución Softmax para Logits: {logits[ejemplo_idx]}")
    plt.ylabel("Probabilidad")
    plt.ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')
        
    plt.savefig("softmax_plot.png", dpi=300)
    print("Gráfico guardado como 'softmax_plot.png'")

if __name__ == "__main__":
    main()