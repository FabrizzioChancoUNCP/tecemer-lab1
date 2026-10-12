import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from datetime import datetime

def graficar_neurona(nombre_activacion, archivo_salida, w=None, b=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    ax.add_patch(plt.Circle((1.5, 2), 0.5, color='skyblue', ec='black', zorder=3))
    ax.text(1.5, 2, 'x\n(Horas)', ha='center', va='center', fontsize=12, zorder=4)

    ax.add_patch(plt.Circle((5, 2), 0.8, color='lightgreen', ec='black', zorder=3))
    ax.text(5, 2, f'Σ\n{nombre_activacion}', ha='center', va='center', fontsize=10, zorder=4)

    ax.add_patch(plt.Circle((8.5, 2), 0.5, color='salmon', ec='black', zorder=3))
    ax.text(8.5, 2, 'y\n(Prob)', ha='center', va='center', fontsize=12, zorder=4)

    ax.annotate('', xy=(4.2, 2), xytext=(2, 2), arrowprops=dict(arrowstyle='->', lw=2))
    if w is not None:
        ax.text(3, 2.3, f'w={w:.2f}', ha='center', fontsize=10, color='blue')
        
    ax.annotate('', xy=(8, 2), xytext=(5.8, 2), arrowprops=dict(arrowstyle='->', lw=2))
    if b is not None:
        ax.text(6.9, 2.3, f'b={b:.2f}', ha='center', fontsize=10, color='red')

    ax.set_title(f'Perceptrón Entrenado con {nombre_activacion}')
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"-> Imagen guardada: '{archivo_salida}'")

def main():
    np.random.seed(42) 
    horas_estudio = np.random.uniform(0, 10, 1000) 
    resultado_examen = (horas_estudio + np.random.normal(0, 1.5, 1000) > 5.0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        horas_estudio, resultado_examen, test_size=0.20, random_state=42
    )

    print("--- 1. DISTRIBUCIÓN DE DATOS ---")
    print(f"Total generados: {len(horas_estudio)}")
    print(f"Para entrenar (80%): {len(X_train)}")
    print(f"Para comprobar (20%): {len(X_test)}\n")

    modelo = keras.Sequential([
        keras.layers.Dense(2, input_shape=(1,), activation='softmax')
    ])
    
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])
    valor=50
    print("--- 2. ENTRENAMIENTO ---")
    print(f"Entrenando la neurona con el 80% de los datos ({valor} iteraciones)...")
    historial = modelo.fit(X_train, y_train, epochs=valor, verbose=0)
    print("¡Entrenamiento finalizado!\n")

    print("--- 3. EVALUACIÓN FINAL ---")
    perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión en el 20% de datos de prueba: {precision * 100:.1f}%\n")

    pesos_finales, sesgo_final = modelo.layers[0].get_weights()
    w_aprendido = pesos_finales[0][0]
    b_aprendido = sesgo_final[0]

    horas_prueba = np.array([2.0, 5.0, 8.0])
    predicciones = modelo.predict(horas_prueba, verbose=0)
    
    print("--- 4. PREDICCIONES DE EJEMPLO ---")
    for horas, prob in zip(horas_prueba, predicciones):
        resultado = "Aprobará" if prob[0] > 0.5 else "Reprobará"
        print(f"Estudió {horas}h -> Probabilidad de éxito: {prob[0]*100:.1f}% ({resultado})")
    print()

    print("--- 5. GENERANDO ARCHIVOS GRÁFICOS ---")
    
    tiempo_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    nombre_neurona = f"perceptron_entrenado_{tiempo_actual}.png"
    graficar_neurona("Sigmoide", nombre_neurona, w=w_aprendido, b=b_aprendido)
    
    plt.figure(figsize=(6, 4))
    plt.plot(historial.history['loss'], label='Error (Pérdida)', color='red', linewidth=2)
    plt.title('Curva de Aprendizaje del Perceptrón')
    plt.xlabel('Épocas (Iteraciones)')
    plt.ylabel('Error (Binary Crossentropy)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    nombre_historial = f"historial_aprendizaje_{tiempo_actual}.png"
    plt.savefig(nombre_historial, dpi=300, bbox_inches='tight')
    print(f"-> Imagen guardada: '{nombre_historial}'")

if __name__ == "__main__":
    main()