import numpy as np
import cv2
import math
from sklearn import linear_model
import streamlit as st
from PIL import Image

# Función para calcular línea desde un punto y un vector unitario
def lineCalc(vx, vy, x0, y0):
    scale = 10
    x1 = x0 + scale * vx
    y1 = y0 + scale * vy
    m = (y1 - y0) / (x1 - x0)
    b = y1 - m * x1
    return m, b

# Función para encontrar el punto de intersección
def lineIntersect(m1, b1, m2, b2):
    a_1 = -m1
    b_1 = 1
    c_1 = b1
    a_2 = -m2
    b_2 = 1
    c_2 = b2

    d = a_1 * b_2 - a_2 * b_1
    dx = c_1 * b_2 - c_2 * b_1
    dy = a_1 * c_2 - a_2 * c_1

    intersectionX = dx / d
    intersectionY = dy / d
    return intersectionX, intersectionY

# Función para procesar la imagen y detectar líneas del cruce
def process(im):
    # Parámetros de procesamiento
    radius = 250
    bw_width = 170

    bxLeft = []
    byLeft = []
    bxbyLeftArray = []
    bxbyRightArray = []
    bxRight = []
    byRight = []
    boundedLeft = []
    boundedRight = []

    # 1. Filtrar el color blanco
    lower = np.array([170, 170, 170])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(im, lower, upper)

    # 2. Erosionar la imagen
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(im.shape[0] / 30), 1))
    erode = cv2.erode(mask, erodeStructure, (-1, -1))

    # 3. Encontrar contornos y dibujar líneas verdes en las franjas blancas
    contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)
        if bw > bw_width:
            # Dibujar los rectángulos de cada franja blanca
            cv2.rectangle(im, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            bxRight.append(bx + bw)
            byRight.append(by)
            bxLeft.append(bx)
            byLeft.append(by)
            bxbyLeftArray.append([bx, by])
            bxbyRightArray.append([bx + bw, by])

    # 4. Calcular el promedio para cada línea
    medianR = np.median(bxbyRightArray, axis=0)
    medianL = np.median(bxbyLeftArray, axis=0)

    # Filtrar puntos cercanos al promedio
    for i in bxbyLeftArray:
        if ((medianL[0] - i[0])**2 + (medianL[1] - i[1])**2) < radius**2:
            boundedLeft.append(i)
    for i in bxbyRightArray:
        if ((medianR[0] - i[0])**2 + (medianR[1] - i[1])**2) < radius**2:
            boundedRight.append(i)

    # 5. Aplicar RANSAC
    bxLeft = np.asarray([p[0] for p in boundedLeft])
    byLeft = np.asarray([p[1] for p in boundedLeft])
    bxRight = np.asarray([p[0] for p in boundedRight])
    byRight = np.asarray([p[1] for p in boundedRight])

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(bxLeft.reshape(-1, 1), byLeft)
    model_ransac.fit(bxRight.reshape(-1, 1), byRight)

    # 6. Calcular intersección de las líneas
    vx, vy, x0, y0 = cv2.fitLine(np.array(boundedLeft), cv2.DIST_L2, 0, 0.01, 0.01)
    vx_R, vy_R, x0_R, y0_R = cv2.fitLine(np.array(boundedRight), cv2.DIST_L2, 0, 0.01, 0.01)
    m_L, b_L = lineCalc(vx, vy, x0, y0)
    m_R, b_R = lineCalc(vx_R, vy_R, x0_R, y0_R)
    intersectionX, intersectionY = lineIntersect(m_R, b_R, m_L, b_L)

    # Dibujar líneas y punto de intersección
    cv2.line(im, (int(x0 - 500 * vx), int(y0 - 500 * vy)), (int(x0 + 500 * vx), int(y0 + 500 * vy)), (255, 0, 0), 2)
    cv2.line(im, (int(x0_R - 500 * vx_R), int(y0_R - 500 * vy_R)), (int(x0_R + 500 * vx_R), int(y0_R + 500 * vy_R)), (255, 0, 0), 2)
    cv2.circle(im, (int(intersectionX), int(intersectionY)), 10, (0, 0, 255), 5)

    return im, intersectionX, intersectionY, (m_L, b_L), (m_R, b_R)

# Coordenadas del área del cruce peatonal (delante de la cámara)
area_cruce = np.array([[669, 209], [98, 751], [1343, 738], [929, 210]], np.int32)

# Cargar y procesar la imagen
im = cv2.imread('C:/Users/guiza/Documents/Python/ProcesamientoIMagenes/data/crucepeatonal.jpeg')
processed_img, intersectionX, intersectionY, (m_L, b_L), (m_R, b_R) = process(im)

# Dibujar el área del cruce peatonal con un polígono
cv2.polylines(im, [area_cruce], isClosed=True, color=(0, 255, 0), thickness=3)

# Convertir la imagen de OpenCV a formato de Streamlit
processed_img_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

# Determinar si el paso es seguro
safe_threshold = 500  # Umbral para considerar el paso seguro
if intersectionY < safe_threshold:
    safety_message = "El paso es seguro."
else:
    safety_message = "No es seguro cruzar."

# Mostrar resultados con Streamlit
st.title("Cruce Peatonal Inteligente")
st.image(processed_img_pil, caption="Cruce Peatonal Procesado", use_column_width=True)
st.subheader(f"Intersección: ({intersectionX[0]:.2f}, {intersectionY[0]:.2f})")
st.subheader(safety_message)
