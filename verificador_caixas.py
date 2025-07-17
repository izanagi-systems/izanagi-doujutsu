import os
# Impede a biblioteca de verificar atualizações online.
os.environ['ULTRALYTICS_SYNC'] = 'False'

import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

class ContadorDeItens:
    def __init__(self, item_model_path='modelos_producao/item_detector.pt', roi_model_path='modelos_producao/roi_detector.pt', perfil_caixa=None, conf_threshold=0.4):
        """
        Inicializa o contador de itens.
        :param item_model_path: Caminho para o modelo YOLO treinado para contar itens.
        :param roi_model_path: Caminho para o modelo YOLO treinado para detectar a caixa principal (ROI).
        :param perfil_caixa: Dicionário com o perfil da caixa (nome, itens_esperados).
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

        # --- Constantes de Desenho ---
        self.CORES = {
            'roi': (255, 0, 255),          # Rosa para ROI
            'item_ok': (0, 255, 0),        # Verde para itens contados
            'divisor': (0, 255, 255),      # Ciano/Amarelo para divisor
            'texto_status': (0, 0, 0),     # Preto para status
            'texto_contagem': (0, 0, 255)  # Vermelho para contagem
        }
        self.FONTE = cv2.FONT_HERSHEY_SIMPLEX
        self.ESCALA_FONTE_INFO = 0.7
        self.ESPESSURA_LINHA = 2

        # --- Máquina de Estados --- 
        self.status_sistema = 'AGUARDANDO_CAIXA' # Estados: AGUARDANDO_CAIXA, CONTANDO_ITENS, AGUARDANDO_DIVISOR, AVALIACAO_FINAL
        self.roi_atual = None

        # --- Controle de Camadas e Contagem ---
        self.camada_atual = 1
        self.contagens_por_camada = [] # Armazena a contagem final de cada camada
        self.contagem_buffer = deque(maxlen=10) # Buffer para suavizar a contagem de itens
        self.divisor_buffer = deque(maxlen=10)  # Buffer para confirmar detecção do divisor

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
            cls = int(box.cls)
            # Desenha o contorno apenas para a classe 'divisor'
            if self.item_model.names[cls] == 'divisor':
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                conf = float(box.conf)
                label = f'divisor {conf:.2f}'
                color = self.CORES['divisor'] # Ciano/Amarelo

                # Desenha o retângulo e o texto
                cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), color, self.ESPESSURA_LINHA)
                cv2.putText(frame_desenhado, label, (x1, y1 - 10), self.FONTE, 0.5, color, self.ESPESSURA_LINHA)

        contagem, roi_desenho, divisores = self._gerenciar_ciclo_de_vida(frame)
        self._desenhar_visualizacoes(frame_desenhado, contagem, roi_desenho, divisores)

        cv2.imshow('SIAC - Verificador de Caixas', frame_desenhado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _loop_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
            return

        paused = False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Fim do vídeo. Reiniciando em 5 segundos...")
                    time.sleep(5)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_desenhado = frame.copy()
                contagem, roi_desenho, divisores = self._gerenciar_ciclo_de_vida(frame)
                self._desenhar_visualizacoes(frame_desenhado, contagem, roi_desenho, divisores)

                # Adiciona informações de navegação no vídeo
                current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.putText(frame_desenhado, f"Frame {current_frame_num}/{total_frames}", (10, 90), self.FONTE, 0.7, (0, 0, 0), self.ESPESSURA_LINHA)

                cv2.imshow('SIAC - Verificador de Caixas', frame_desenhado)

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
                ret, frame = cap.read()
            elif key == ord('a'):
                pos_atual = cap.get(cv2.CAP_PROP_POS_FRAMES)
                pos_anterior = max(0, pos_atual - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_anterior)
                ret, frame = cap.read()
            
            if not paused:
                ret, frame = cap.read()

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
                print("[ERRO] Não foi possível capturar o frame da câmera.")
                break

            frame_desenhado = frame.copy()
            contagem, roi_desenho, divisores = self._gerenciar_ciclo_de_vida(frame)
            self._desenhar_visualizacoes(frame_desenhado, contagem, roi_desenho, divisores)

            cv2.imshow('SIAC - Verificador de Caixas', frame_desenhado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _gerenciar_ciclo_de_vida(self, frame):
        """
        Gerencia o estado da detecção da caixa (ROI) e a contagem de itens.
        Retorna a contagem de itens, as coordenadas da ROI e os divisores detectados.
        """
        # Etapa 1: Detectar TODOS os objetos no frame inteiro, uma única vez.
        results = self.item_model(frame, verbose=False, conf=self.conf_threshold)
        todos_itens = []
        todos_divisores = []
        for box in results[0].boxes:
            coords = [int(i) for i in box.xyxy[0]]
            if int(box.cls) == 0: # item
                todos_itens.append(coords)
            elif int(box.cls) == 1: # divisor
                todos_divisores.append(coords)

        # Etapa 2: Detectar a ROI (caixa principal) consistentemente
        roi_detectada = self._detectar_roi_estavel(frame)
        contagem_display = 0

        # --- Início da Máquina de Estados ---

        if self.status_sistema == 'AGUARDANDO_CAIXA':
            if roi_detectada:
                print("[STATUS] Caixa detectada. Iniciando contagem da camada 1...")
                self.status_sistema = 'CONTANDO_ITENS'
                self.roi_atual = roi_detectada
                self.camada_atual = 1
                self.contagens_por_camada = []
                self.contagem_buffer.clear()
                self.divisor_buffer.clear()

        elif self.status_sistema == 'CONTANDO_ITENS':
            # Se a caixa ainda está visível, continua o processo
            if roi_detectada:
                self.roi_atual = roi_detectada # Atualiza a posição da ROI

                # Filtra os itens que estão DENTRO da ROI atual
                itens_na_roi = []
                rx1, ry1, rx2, ry2 = self.roi_atual
                for item_box in todos_itens:
                    ix1, iy1, ix2, iy2 = item_box
                    # Usa o centro do item para verificar se está na ROI
                    centro_x = (ix1 + ix2) / 2
                    centro_y = (iy1 + iy2) / 2
                    if rx1 < centro_x < rx2 and ry1 < centro_y < ry2:
                        itens_na_roi.append(item_box)
                
                self.contagem_buffer.append(len(itens_na_roi))

            else: # Caixa foi perdida
                print("[STATUS] Caixa removida. Aguardando nova caixa...")
                self.status_sistema = 'AGUARDANDO_CAIXA'
                self.roi_atual = None

            # Usa a média do buffer para estabilizar a contagem no display
            if self.contagem_buffer:
                contagem_display = int(np.mean(self.contagem_buffer))

        elif self.status_sistema == 'AGUARDANDO_DIVISOR':
            contagem_display = self.contagens_por_camada[-1] if self.contagens_por_camada else 0

        # Retorna os dados para a função de desenho
        return contagem_display, self.roi_atual, todos_divisores

    def _detectar_roi_estavel(self, frame):
        """
        Usa o modelo YOLO para detectar a ROI. 
        Retorna as coordenadas da maior caixa detectada, ou None se nenhuma for encontrada.
        """
        roi_results = self.roi_model(frame, verbose=False, conf=0.5)
        caixas_detectadas = roi_results[0].boxes.xyxy

        if len(caixas_detectadas) > 0:
            # Pega a maior caixa detectada como a ROI principal
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in caixas_detectadas]
            maior_caixa_idx = np.argmax(areas)
            x1, y1, x2, y2 = caixas_detectadas[maior_caixa_idx]
            return [int(x1), int(y1), int(x2), int(y2)]
        
        return None

    def _desenhar_visualizacoes(self, frame, contagem, roi_desenho, divisores):
        """
        Desenha todas as informações visuais no frame: ROI, contagem, status e divisores.
        """
        # Desenha a ROI
        if roi_desenho:
            x1, y1, x2, y2 = roi_desenho
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.CORES['roi'], self.ESPESSURA_LINHA) # Cor Rosa/Magenta
            cv2.putText(frame, "ROI Caixa", (x1, y1 - 10), 
                        self.FONTE, self.ESCALA_FONTE_INFO, self.CORES['roi'], self.ESPESSURA_LINHA)

        # Desenha o status do sistema e a contagem
        status_texto = f"Status: {self.status_sistema}"
        contagem_texto = f"Itens Camada {self.camada_atual}: {contagem}"
        cv2.putText(frame, status_texto, (10, 30), self.FONTE, self.ESCALA_FONTE_INFO, self.CORES['texto_status'], self.ESPESSURA_LINHA)
        cv2.putText(frame, contagem_texto, (10, 60), self.FONTE, self.ESCALA_FONTE_INFO, self.CORES['texto_contagem'], self.ESPESSURA_LINHA)

        # Desenha os divisores
        for d in divisores:
            x1, y1, x2, y2 = d
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.CORES['divisor'], self.ESPESSURA_LINHA) # Cor Ciano/Amarelo
            cv2.putText(frame, 'divisor', (x1, y1 - 10), self.FONTE, 0.5, self.CORES['divisor'], self.ESPESSURA_LINHA)

if __name__ == '__main__':
    # --- CONFIGURAÇÕES ---
    # Define o perfil da caixa (o que se espera que ela contenha)
    perfil = {
        "nome": "Caixa Padrão",
        "itens_esperados": 12,
        "total_camadas": 3
    }

    # --- INICIALIZAÇÃO E EXECUÇÃO ---
    contador = ContadorDeItens(
        item_model_path='modelos_producao/item_detector.pt',
        roi_model_path='modelos_producao/roi_detector.pt',
        perfil_caixa=perfil,
        conf_threshold=0.4
    )

    try:
        contador.run(video_source=0)
    finally:
        # Garante que todas as janelas do OpenCV sejam fechadas ao final
        cv2.destroyAllWindows()