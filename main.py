# import cv2
# import tkinter as tk
# from threading import Thread
# import winsound

# # Carregar o classificador pré-treinado de pessoas do OpenCV
# HOGCV = cv2.HOGDescriptor()
# HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# def detect_people():
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Ler o quadro atual da webcam
#         r, frame = cap.read()
#         if r:
#             # Redimensionar o quadro
#             frame = cv2.resize(frame, (640, 480))

#             # Detectar pessoas no quadro
#             boxes, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

#             # Desenhar caixas delimitadoras ao redor das pessoas detectadas
#             for (x, y, w, h) in boxes:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 # Tocar um bipe quando uma pessoa é detectada
#                 # winsound.Beep(1000, 500)

#             # Mostrar o quadro
#             cv2.imshow("preview", frame)

#         # Sair se a tecla ESC for pressionada
#         if cv2.waitKey(1) == 27:
#             break

#     # Liberar recursos e fechar janelas
#     cap.release()
#     cv2.destroyAllWindows()

# # Criar a janela principal
# root = tk.Tk()

# # Adicionar um botão à janela
# button = tk.Button(root, text="Iniciar detecção de pessoas", command=lambda: Thread(target=detect_people).start())
# button.pack()

# # Iniciar o loop principal da interface de usuário
# root.mainloop()


import cv2
import numpy as np
from openpose import pyopenpose as op

# Configuração do OpenPose
params = dict()
params["model_folder"] = "/poses"

# Inicializando o OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Processando a imagem com o OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Desenhando os pontos chave do corpo na imagem
    image = datum.cvOutputData

    cv2.imshow("OpenPose", image)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()