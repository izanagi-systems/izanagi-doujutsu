import os
# SOLUÇÃO DEFINITIVA PARA ERRO DE REDE: Impede a biblioteca de verificar atualizações online.
os.environ['ULTRALYTICS_SYNC'] = 'False'

import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import deque
import argparse

class ContadorDeItens:
    def __init__(self, item_model_path='runs/detect/contador_multiclasse_final/weights/best.pt', roi_model_path='path/to/roi_model.pt', perfil_caixa=None, timeout_alarme=50, conf_threshold=0.4):
        """
        Inicializa o contador de itens.
        :param item_model_path: Caminho para o modelo YOLO treinado para contar itens.
        :param roi_model_path: Caminho para o modelo YOLO treinado para detectar a caixa principal (ROI).
        :param perfil_caixa: Dicionário com o perfil da caixa (nome, itens_esperados).
        :param timeout_alarme: Frames para esperar antes de alarmar.
        :param conf_threshold: Limite de confiança para detecções de itens.
        """
        print(f"[INFO] Carregando modelos...")
        self.item_model = YOLO(item_model_path)
        self.roi_model = YOLO(roi_model_path)
        print(f"[INFO] Modelos carregados com sucesso.")

        self.perfil_caixa = perfil_caixa
        self.conf_threshold = conf_threshold # Armazena o limite de confiança
        if self.perfil_caixa:
            print(f"[INFO] Perfil carregado: {self.perfil_caixa['nome']} (Esperando {self.perfil_caixa['itens_esperados']} itens por camada para {self.perfil_caixa['total_camadas']} camadas)")
        else:
            print("[AVISO] Nenhum perfil de caixa carregado. A contagem será apenas exibida.")

        # --- Estados do Sistema ---
        self.status_caixa = 'AGUARDANDO_CAIXA'
        self.roi_atual = None
        self.roi_travada = False
        self.frames_sem_roi = 0
        self.timeout_sem_roi = timeout_alarme # Frames para esperar antes de resetar
        self.contagem_buffer = deque(maxlen=15) # Buffer para suavizar a contagem
        self.camada_atual = 1

    def run(self, imagem_path=None, video_source=0):
        if imagem_path:
            print(f"[INFO] Processando imagem única: {imagem_path}")
            self._processar_imagem_unica(imagem_path)
        elif isinstance(video_source, str): # Caminho para arquivo de vídeo
            print("[INFO] Iniciando modo Arquivo de Vídeo.")
            self._loop_video_file(video_source)
        else: # Câmera ao vivo
            print("[INFO] Iniciando modo Câmera ao Vivo. Pressione 'q' para sair.")
            self._loop_live_camera(video_source)

    def _processar_imagem_unica(self, imagem_path):
        """
        Processa uma única imagem, realiza a detecção e exibe o resultado.
        """
        frame = cv2.imread(imagem_path)
        if frame is None:
            print(f"[ERRO] Não foi possível carregar a imagem: {imagem_path}")
            return

        # Roda a detecção
        results = self.item_model(frame, verbose=False, conf=self.conf_threshold)
        frame_desenhado = frame.copy() # Cria uma cópia para desenhar

        # Loop manual para desenhar cada detecção
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            conf = float(box.conf)
            cls = int(box.cls)
            label = f'{self.item_model.names[cls]} {conf:.2f}'
            
            # Define a cor baseada na classe (item = azul, divisor = ciano/amarelo)
            color = (255, 0, 0) if cls == 0 else (0, 255, 255)

            # Desenha o retângulo e o texto
            cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_desenhado, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        contagem, roi_desenho = self._gerenciar_ciclo_de_vida(frame)
        self._desenhar_visualizacoes(frame_desenhado, contagem, roi_desenho)

        cv2.imshow('SIAC - Verificador de Caixas', frame_desenhado)

    def _gerenciar_ciclo_de_vida(self, frame):
        """
        Gerencia o estado da detecção da caixa (ROI) e a contagem de itens.
        Retorna a contagem de itens e as coordenadas da ROI para desenho.
        """
        # 1. Detectar a ROI (caixa principal)
        roi_results = self.roi_model(frame, verbose=False, conf=0.5)
        caixas_detectadas = roi_results[0].boxes.xyxy
        
        roi_para_desenho = None
        contagem_final = 0

        if len(caixas_detectadas) > 0:
            # Pega a maior caixa detectada como a ROI principal
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in caixas_detectadas]
            maior_caixa_idx = np.argmax(areas)
            x1, y1, x2, y2 = caixas_detectadas[maior_caixa_idx]
            self.roi_atual = [int(x1), int(y1), int(x2), int(y2)]
            roi_para_desenho = self.roi_atual
            self.frames_sem_roi = 0
            self.status_caixa = 'CAIXA_PRESENTE'

        else: # Nenhuma caixa detectada
            self.frames_sem_roi += 1
            if self.frames_sem_roi > self.timeout_sem_roi:
                # Se a caixa sumiu por tempo suficiente, reseta o estado
                self.status_caixa = 'AGUARDANDO_CAIXA'
                self.roi_atual = None
                self.contagem_buffer.clear()
                self.camada_atual = 1

        # 2. Se a caixa está presente, conta os itens dentro dela
        if self.status_caixa == 'CAIXA_PRESENTE' and self.roi_atual:
            x1, y1, x2, y2 = self.roi_atual
            roi_frame = frame[y1:y2, x1:x2]

            if roi_frame.size > 0:
                item_results = self.item_model(roi_frame, verbose=False, conf=self.conf_threshold)
                # Filtra apenas a classe 'item' (ID 0)
                contagem_itens = sum(1 for box in item_results[0].boxes if int(box.cls) == 0)
                self.contagem_buffer.append(contagem_itens)
            
            # Usa a média do buffer para estabilizar a contagem
            if self.contagem_buffer:
                contagem_final = int(np.mean(self.contagem_buffer))
            else:
                contagem_final = 0
        
        return contagem_final, roi_para_desenho

    def _desenhar_visualizacoes(self, frame, contagem, roi_desenho):
        """
        Desenha as informações de contagem e ROI no frame.
{{ ... }}
            cv2.putText(frame, "ROI Caixa", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        """
    def _loop_video_file(self, video_source):
        """
        Loop de processamento para arquivos de vídeo com controles de player.
        Espaço: Play/Pause. 'd'/'a': Frame a frame (quando pausado). 'q': Sair.
        """
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print(f"[ERRO] Não foi possível abrir o arquivo de vídeo: {video_source}")
            return

        paused = True
        ret, frame = self.cap.read()

        while True:
            if ret:
                # Roda a detecção e desenha na tela
                results = self.item_model(frame, verbose=False, conf=self.conf_threshold)
                frame_desenhado = frame.copy() # Cria uma cópia para desenhar

                # Loop manual para desenhar cada detecção
                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    conf = float(box.conf)
                    cls = int(box.cls)
                    label = f'{self.item_model.names[cls]} {conf:.2f}'
                    
                    # Define a cor baseada na classe (item = azul, divisor = ciano/amarelo)
                    color = (255, 0, 0) if cls == 0 else (0, 255, 255)

                    # Desenha o retângulo e o texto
                    cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_desenhado, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                contagem, roi_desenho = self._gerenciar_ciclo_de_vida(frame)
                self._desenhar_visualizacoes(frame_desenhado, contagem, roi_desenho)

                cv2.imshow('SIAC - Verificador de Caixas', frame_desenhado)
            else:
                print("[AVISO] Fim do vídeo ou erro na captura.")
                paused = True # Força a pausa no final

            # Lógica de controle de vídeo
            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d') and paused:
                ret, frame = self.cap.read()
            elif key == ord('a'):
                pos_atual = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                pos_anterior = max(0, pos_atual - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos_anterior)
                ret, frame = self.cap.read()
            
            if not paused:
                ret, frame = self.cap.read()

    def _loop_live_camera(self, video_source):
        """
        Loop de processamento para câmera ao vivo.
        'q': Sair.
        """
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("[ERRO] Não foi possível abrir a fonte de vídeo.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERRO] Não foi possível ler o frame da câmera. Encerrando.")
                break

            # Roda a detecção e desenha na tela
            results = self.item_model(frame, verbose=False, conf=self.conf_threshold)
            frame_desenhado = frame.copy() # Cria uma cópia para desenhar

            # Loop manual para desenhar cada detecção
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                conf = float(box.conf)
                cls = int(box.cls)
                label = f'{self.item_model.names[cls]} {conf:.2f}'
                
                # Define a cor baseada na classe (item = azul, divisor = ciano/amarelo)
                color = (255, 0, 0) if cls == 0 else (0, 255, 255)

                # Desenha o retângulo e o texto
                cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_desenhado, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            contagem, roi_desenho = self._gerenciar_ciclo_de_vida(frame)
            self._desenhar_visualizacoes(frame_desenhado, contagem, roi_desenho)

            cv2.imshow('SIAC - Verificador de Caixas', frame_desenhado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    # --- CONFIGURAÇÕES ---
    USAR_CAMERA = True
    CAMINHO_VIDEO = 'videos/video_teste_contagem_02.mp4'
    # Para testar o modelo em uma imagem específica do dataset
    CAMINHO_IMAGEM = None
    # Limite de confiança para considerar uma detecção válida
    LIMITE_CONFIANCA = 0.4

    # Perfil da caixa a ser verificado
    perfil = {
        "nome": "Caixa Padrão",
        "itens_esperados": 12,
        "total_camadas": 3
    }

    # --- INICIALIZAÇÃO E EXECUÇÃO ---
    contador = ContadorDeItens(
        item_model_path='runs/detect/modelo_producao_v1/weights/best.pt',
        roi_model_path='runs/detect/roi_detector_final2/weights/best.pt', # Atualize se necessário
        perfil_caixa=perfil,
        conf_threshold=LIMITE_CONFIANCA
    )

    try:
        if USAR_CAMERA:
            contador.run(video_source=0)
        elif CAMINHO_IMAGEM:
            contador.run(imagem_path=CAMINHO_IMAGEM)
            # Se processamos uma imagem, precisamos esperar o usuário fechar a janela
            print("\nPressione qualquer tecla na janela da imagem para sair.")
            cv2.waitKey(0)
        else:
            # Se não for câmera e não for imagem, assume que é vídeo
            contador.run(video_source=CAMINHO_VIDEO)
    finally:
        # Garante que todas as janelas do OpenCV sejam fechadas ao final
        cv2.destroyAllWindows()