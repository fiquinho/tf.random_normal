import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import tensorflow as tf


def histograms_x2(tensor1, tensor2):
    """
    Crea los histogramas de las listas pasadas,
    con anotaciones y líneas marcadas.

    :param tensor1: Primera lista
    :param tensor2: Segunda lista
    :return: None
    """

    # Promedio de los valores de las listas pasadas
    promedio_valores_1 = promedio_valores(tensor1)
    promedio_valores_2 = promedio_valores(tensor2)

    # Crea la figura que va a contener ambos histogramas
    plt.figure(figsize=(15, 5))

    # Figura número 1
    plt.subplot(121)
    cantidad_valores_por_clase_1, clases_1, patches1 = plt.hist(tensor1)
    promedio_cantidades = promedio_valores(cantidad_valores_por_clase_1)

    pyl.axvline(promedio_valores_1, color='r', linewidth=3, linestyle="--")
    pyl.axhline(promedio_cantidades, color='#db9723', linewidth=3, linestyle="--")
    plt.xlabel('Magnitud de los valores')
    plt.ylabel('Cantidad de valores por clase')
    plt.title('Histograma {} valores'.format(len(tensor1)))
    plt.text(promedio_valores_1 * 1.05, max(cantidad_valores_por_clase_1) * .9,
             "Promedio valores \naleatorios creados:\n{}".format(round(promedio_valores_1, 5)),
             bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 2})
    plt.text(clases_1[0], promedio_cantidades * .8,
             "Promedio de cantidad\nde valores por clase:\n{}".format(promedio_cantidades),
             bbox={'facecolor': 'orange', 'alpha': 0.75, 'pad': 2})
    plt.grid()

    # Figura número 2
    plt.subplot(122)
    cantidad_valores_por_clase_2, clases_2, patches2 = plt.hist(tensor2)
    cantidad_máxima_2 = max(cantidad_valores_por_clase_2)
    promedio_cantidades = promedio_valores(cantidad_valores_por_clase_2)
    pyl.axvline(promedio_valores_2, color='r', linewidth=3, linestyle="--")
    pyl.axhline(promedio_cantidades, color='#db9723', linewidth=3, linestyle="--")

    plt.xlabel('Magnitud de los valores')
    plt.ylabel('Cantidad de valores por clase')
    plt.title('Histograma {} valores'.format(len(tensor2)))
    plt.text(promedio_valores_2 * 1.05, cantidad_máxima_2 * .9,
             "Promedio valores \naleatorios creados:\n{}".format(round(promedio_valores_2, 5)),
             bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 2})
    plt.text(clases_2[0], promedio_cantidades * 0.81,
             "Promedio de cantidad\nde valores por clase:\n{}".format(promedio_cantidades),
             bbox={'facecolor': 'orange', 'alpha': 0.75, 'pad': 2})
    plt.grid()

    plt.show()


def promedio_valores(tensor):
    """
    Calcula el promedio de los valores de la lista pasada

    :param tensor: La lista con los valores
    :return: El promedio de los valores
    """
    total = 0
    for x in tensor:
        total += x
    promedio = total / len(tensor)
    return promedio


# Creación de ambos tensores para test
tensor_100 = tf.random_normal([100])
tensor_1000 = tf.random_normal([1000], mean=20, stddev=2)

# Generación de los valores aleatorios
with tf.Session() as sess:
    tensor_100 = sess.run(tensor_100)
    tensor_1000 = sess.run(tensor_1000)

# Función de prueba
histograms_x2(tensor_100, tensor_1000)
