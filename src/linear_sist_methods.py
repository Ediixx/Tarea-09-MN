# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np
from sympy import symbols, Eq, solve, Matrix


# ####################################################################
def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            logging.debug(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

        print(f"\n{A}")
    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
        solucion[i] = (A[i, n] - suma) / A[i, i]

    return solucion

# ####################################################################

def eliminacion_gaussiana_pivoteo (A: np.ndarray) -> np.ndarray:
    A = np.array(A)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi


        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            print(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]


    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    print(f"\n{A}")
# --- Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
        solucion[i] = (A[i, n] - suma) / A[i, i]

    return solucion

# # ####################################################################

def eliminacion_gaussiana_punto_flotante (A: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi


        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            print(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    # --- Sustitución hacia atrás
    solucion = np.zeros(n, dtype=np.float32)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
        solucion[i] = (A[i, n] - suma) / A[i, i]
    return solucion


# ####################################################################

def Gauss_Jordan_punto_flotante(A: np.ndarray) -> np.ndarray:

    # Copiar la matriz para evitar modificar el original
    A = A.astype(float)
    rows, cols = A.shape
    
    for i in range(rows):
        # Hacer el pivote igual a 1
        pivot = A[i, i]
        if pivot == 0:
            raise ValueError(f"El pivote en la fila {i+1} es cero. Reordenar filas es necesario.")
        A[i] = A[i] / pivot
        print(f"Paso {i+1}: Normalizar fila {i+1} (hacer pivote igual a 1):")
        print(A)
        print("-" * 50)

        # Hacer ceros en las demás filas de la columna del pivote
        for j in range(rows):
            if j != i:
                factor = A[j, i]
                A[j] = A[j] - factor * A[i]
                print(f"Paso {i+1}.{j+1}: Hacer cero el elemento en la posición ({j+1}, {i+1}):")
                print(A)
                print("-" * 50)
    
    # Retornar las soluciones (última columna)
    soluciones = A[:, -1]
    return soluciones

# ####################################################################

def analizar_sistema(matriz):
    # Declarar alpha como símbolo
    alpha = symbols('alpha')

    # Separar la matriz en coeficientes (A) y términos independientes (b)
    A = Matrix([fila[:-1] for fila in matriz])
    b = Matrix([fila[-1] for fila in matriz])

    # Determinante de la matriz de coeficientes
    det_A = A.det()

    # a. Valores de alpha para los cuales no hay solución (det(A) = 0)
    alpha_no_sol = solve(det_A, alpha)

    # b. Determinar si el rango de la matriz aumentada coincide con el rango de la matriz A
    valores_infinitas_sol = []
    for a in alpha_no_sol:
        A_a = A.subs(alpha, a)
        b_a = b
        aug_matrix = A_a.row_join(b_a)
        if A_a.rank() == aug_matrix.rank() and A_a.rank() < A.rows:
            valores_infinitas_sol.append(a)

    # c. Encontrar la solución única para un valor específico de alpha
    solucion_unica = None
    if len(alpha_no_sol) == 0 or all(A.subs(alpha, a).det() != 0 for a in alpha_no_sol):
        # Si alpha_no_sol está vacío o no anula el determinante
        A_num = A.subs(alpha, 0)  # Por ejemplo, usar alpha = 0 si es válido
        b_num = b
        solucion_unica = np.linalg.solve(np.array(A_num).astype(float), np.array(b_num).astype(float))

    # Retornar resultados
    return {
        "No existe solución en: ": alpha_no_sol,
        "Soluciones infinitas en: ": valores_infinitas_sol,
        "Solución Única en: ": solucion_unica
    }

# ####################################################################

def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A.
    [IMPORTANTE] No se realiza pivoteo.

    ## Parameters

    ``A``: matriz cuadrada de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior. Se obtiene de la matriz ``A`` después de aplicar la eliminación gaussiana.
    """

    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero

    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.zeros((n, n), dtype=float)

    for i in range(0, n):  # loop por columna

        # --- deterimnar pivote
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")

        # --- Eliminación: loop por fila
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

            L[j, i] = m

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    return L, A


# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU.

    ## Parameters

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior.

    ``b``: vector de términos independientes.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """

    n = L.shape[0]

    # --- Sustitución hacia adelante
    logging.info("Sustitución hacia adelante")

    y = np.zeros((n, 1), dtype=float)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i]

    logging.info(f"y = \n{y}")

    # --- Sustitución hacia atrás
    logging.info("Sustitución hacia atrás")
    sol = np.zeros((n, 1), dtype=float)

    sol[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        logging.info(f"i = {i}")
        suma = 0
        for j in range(i + 1, n):
            suma += U[i, j] * sol[j]
        logging.info(f"suma = {suma}")
        logging.info(f"U[i, i] = {U[i, i]}")
        logging.info(f"y[i] = {y[i]}")
        sol[i] = (y[i] - suma) / U[i, i]

    logging.debug(f"x = \n{sol}")
    return sol


# ####################################################################
def matriz_aumentada(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Construye la matriz aumentada de un sistema de ecuaciones lineales.

    ## Parameters

    ``A``: matriz de coeficientes.

    ``b``: vector de términos independientes.

    ## Return

    ``Ab``: matriz aumentada.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert A.shape[0] == b.shape[0], "Las dimensiones de A y b no coinciden."
    return np.hstack((A, b.reshape(-1, 1)))


# ####################################################################
def separar_m_aumentada(Ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa la matriz aumentada en la matriz de coeficientes y el vector de términos independientes.

    ## Parameters
    ``Ab``: matriz aumentada.

    ## Return
    ``A``: matriz de coeficientes.
    ``b``: vector de términos independientes.
    """
    logging.debug(f"Ab = \n{Ab}")
    if not isinstance(Ab, np.ndarray):
        logging.debug("Convirtiendo Ab a numpy array")
        Ab = np.array(Ab, dtype=float)
    return Ab[:, :-1], Ab[:, -1].reshape(-1, 1)
