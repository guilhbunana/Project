import cv2
import numpy as np
import datetime

# Fator de conversão: 1 pixel = 0.5 mm (ajuste conforme sua calibração real)
SCALE_MM_PER_PIXEL = 0.5

# Altura da mesa em mm (eixo Z), se o braço pegar sobre a superfície
Z_FIXED_MM = 0

# Nome do arquivo de saída
output_file = "coordenadas_para_braco.txt"

# Abre a câmera (use 0 ou 1 dependendo da sua webcam)
cap = cv2.VideoCapture(0)

# Abre o arquivo de saída
with open(output_file, 'w') as file:
    file.write("X_mm,Y_mm,Z_mm,timestamp\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)

                # Coordenadas do centro do objeto em pixels
                cx_pixel = x + w // 2
                cy_pixel = y + h // 2

                # Conversão para milímetros
                x_mm = cx_pixel * SCALE_MM_PER_PIXEL
                y_mm = cy_pixel * SCALE_MM_PER_PIXEL
                z_mm = Z_FIXED_MM

                # Timestamp atual
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Escreve no arquivo
                file.write(f"{x_mm:.2f},{y_mm:.2f},{z_mm},{timestamp}\n")

                # Desenha o retângulo e centro na tela
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{x_mm:.1f} mm, {y_mm:.1f} mm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Deteccao para braco", frame)

        # Pressione Q para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
