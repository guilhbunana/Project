import cv2
import numpy as np

# Abre a câmera padrão (geralmente a webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona o frame (opcional, para melhorar performance)
    frame = cv2.resize(frame, (640, 480))

    # Converte para escala de cinza e aplica um desfoque
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecta bordas
    edges = cv2.Canny(blur, 50, 150)

    # Encontra contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos por tamanho (ajuste os valores conforme sua cena)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # você pode ajustar esse valor
            x, y, w, h = cv2.boundingRect(cnt)
            # Exibe os contornos como objetos detectados
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibe a imagem final
    cv2.imshow("Objetos sobre a mesa", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
