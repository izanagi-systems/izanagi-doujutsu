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
        self.status_sistema = 'AGUARDANDO_CAIXA' # Estados: AGUARDANDO_CAIXA, CONTANDO_ITENS, AGUARDANDO_DIVISOR, CAIXA_COMPLETA, ERRO_DIVISOR_PRECOCE, CAIXA_AUSENTE
        self.estado_anterior = None # Guarda o último estado antes de pausar
        self.roi_atual = None
        self.roi_buffer = deque(maxlen=5) # Buffer para estabilizar a detecção da ROI
        self.divisor_buffer = deque(maxlen=5) # Buffer para estabilizar a detecção do divisor

        # --- Controle de Camadas e Contagem ---
        self.camada_atual = 1
        self.contagens_por_camada = [] # Armazena a contagem final de cada camada
        self.contagem_buffer = deque(maxlen=15) # Buffer para suavizar a contagem de itens

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
        deteccoes_roi = self.roi_model.predict(source=frame, conf=self.conf_threshold, verbose=False)[0]
        deteccoes_itens_divisores = self.item_model.predict(source=frame, conf=self.conf_threshold, verbose=False)[0]

        # Atualiza o buffer da ROI e determina se a ROI está estável
        roi_detectada = self._extrair_melhor_roi(deteccoes_roi.boxes)
        self.roi_buffer.append(1 if roi_detectada else 0)
        # Considera a ROI estável se foi detectada em mais da metade dos frames do buffer
        roi_estavel = sum(self.roi_buffer) > (self.roi_buffer.maxlen / 2)

        # Separa as detecções de itens e divisores
        todos_itens, todos_divisores = self._separar_itens_e_divisores(deteccoes_itens_divisores.boxes)

        contagem_display = 0

        # Lógica da máquina de estados principal
        if self.status_sistema == 'AGUARDANDO_CAIXA':
            if roi_estavel:
                print("[STATUS] Caixa detectada. Iniciando contagem...")
                self.status_sistema = 'CONTANDO_ITENS'
                self.roi_atual = roi_detectada
                # Limpa tudo para um novo ciclo
                self._reset_ciclo()

        elif self.status_sistema == 'CAIXA_AUSENTE':
            if roi_estavel:
                print(f"[STATUS] Caixa retornou. Retomando do estado: {self.estado_anterior}")
                self.status_sistema = self.estado_anterior
                self.estado_anterior = None
                # Não limpa o roi_buffer para manter a estabilidade
                self.divisor_buffer.clear() # Limpa para reavaliar o divisor na retomada

        elif self.status_sistema == 'CONTANDO_ITENS':
            # Se a caixa ainda está visível, continua o processo
            if roi_estavel:
                self.roi_atual = roi_detectada or self.roi_atual # Atualiza a posição da ROI se detectada

                # Filtra os itens que estão DENTRO da ROI atual
                itens_na_roi = self._filtrar_objetos_na_roi(todos_itens, self.roi_atual)
                self.contagem_buffer.append(len(itens_na_roi))
                contagem_estavel = max(set(list(self.contagem_buffer)), key=list(self.contagem_buffer).count) if self.contagem_buffer else 0

                # Verifica a presença de divisor ANTES da hora
                divisores_na_roi = self._filtrar_objetos_na_roi(todos_divisores, self.roi_atual)
                self.divisor_buffer.append(1 if divisores_na_roi else 0)
                divisor_estavel_presente = sum(self.divisor_buffer) > (self.divisor_buffer.maxlen / 2)

                # A verificação de divisor precoce SÓ deve acontecer na primeira camada e se já houver itens.
                if self.camada_atual == 1 and divisor_estavel_presente and contagem_estavel > 0:
                    print("[ALERTA] Divisor detectado antes da camada estar completa! Remova o divisor.")
                    self.status_sistema = 'ERRO_DIVISOR_PRECOCE'

                # Lógica de contagem e transição de camada (independente do erro acima)
                elif self.perfil_caixa and contagem_estavel >= self.perfil_caixa['itens_esperados']:
                    print(f"[STATUS] Camada {self.camada_atual} completa com {contagem_estavel} itens.")
                    self.contagens_por_camada.append(contagem_estavel)
                    
                    # Decide para qual estado ir a seguir
                    if self.camada_atual < self.perfil_caixa['total_camadas']:
                        self.status_sistema = 'AGUARDANDO_DIVISOR'
                    else:
                        print("[STATUS] Caixa finalizada! Todas as camadas estão corretas.")
                        self.status_sistema = 'CAIXA_COMPLETA'
                    
                    self.contagem_buffer.clear() # Limpa o buffer para a próxima camada

            else: # Caixa foi perdida
                print("[STATUS] Caixa removida durante a contagem. Pausando...")
                self.estado_anterior = 'CONTANDO_ITENS'
                self.status_sistema = 'CAIXA_AUSENTE'
                self.roi_atual = None

        elif self.status_sistema == 'AGUARDANDO_DIVISOR':
            if roi_estavel:
                self.roi_atual = roi_detectada or self.roi_atual # Mantém o rastreamento da caixa

                # Foca na detecção do divisor para avançar para a próxima camada
                divisores_na_roi = self._filtrar_objetos_na_roi(todos_divisores, self.roi_atual)
                self.divisor_buffer.append(1 if divisores_na_roi else 0)
                divisor_estavel_presente = sum(self.divisor_buffer) > (self.divisor_buffer.maxlen / 2)

                if divisor_estavel_presente:
                    # Verifica se a caixa já está completa
                    if self.camada_atual >= self.perfil_caixa['total_camadas']:
                        print("[STATUS] Caixa finalizada! Todas as camadas e divisores estão corretos.")
                        self.status_sistema = 'CAIXA_COMPLETA'
                    else:
                        # Avança para a próxima camada
                        print(f"[STATUS] Divisor confirmado. Iniciando contagem da camada {self.camada_atual + 1}...")
                        self.camada_atual += 1
                        self.status_sistema = 'CONTANDO_ITENS'
                        self.contagem_buffer.clear() # Limpa o buffer para a nova camada
                        self.divisor_buffer.clear() # ESSENCIAL: Reseta a detecção para a nova camada
            else:
                print("[STATUS] Caixa removida enquanto aguardava divisor. Pausando...")
                self.estado_anterior = 'AGUARDANDO_DIVISOR'
                self.status_sistema = 'CAIXA_AUSENTE'
                self.roi_atual = None

        elif self.status_sistema == 'ERRO_DIVISOR_PRECOCE':
            if roi_estavel:
                # Aguarda o divisor ser removido
                divisores_na_roi = self._filtrar_objetos_na_roi(todos_divisores, self.roi_atual)
                self.divisor_buffer.append(1 if divisores_na_roi else 0)
                divisor_estavel_presente = sum(self.divisor_buffer) > (self.divisor_buffer.maxlen / 2)

                if not divisor_estavel_presente:
                    print("[STATUS] Divisor removido. Retomando contagem...")
                    self.status_sistema = 'CONTANDO_ITENS'
            else:
                print("[STATUS] Caixa removida durante alerta de divisor. Pausando...")
                self.estado_anterior = 'ERRO_DIVISOR_PRECOCE'
                self.status_sistema = 'CAIXA_AUSENTE'
                self.roi_atual = None

        elif self.status_sistema == 'CAIXA_COMPLETA':
            # Aguarda a caixa ser removida para reiniciar o ciclo
            if not roi_estavel:
                print("[STATUS] Caixa completa removida. Sistema pronto para novo ciclo.")
                self.status_sistema = 'AGUARDANDO_CAIXA'
                self.roi_atual = None

        # Usa a média do buffer para estabilizar a contagem no display
        if self.contagem_buffer:
            contagem_display = max(set(list(self.contagem_buffer)), key=list(self.contagem_buffer).count)

        # Retorna os dados para a função de desenho
        return contagem_display, self.roi_atual, todos_divisores

    def _reset_ciclo(self):
        """Reseta as variáveis de controle para um novo ciclo de contagem."""
        print("[INFO] Resetando ciclo...")
        self.camada_atual = 1
        self.contagens_por_camada.clear()
        self.contagem_buffer.clear()
        self.divisor_buffer.clear()
        self.roi_atual = None
        self.estado_anterior = None

    def _extrair_melhor_roi(self, deteccoes_roi):
        """
        Usa o modelo YOLO para detectar a ROI. 
        Retorna as coordenadas da maior caixa detectada, ou None se nenhuma for encontrada.
        """
        coordenadas_roi = deteccoes_roi.xyxy
        if len(coordenadas_roi) > 0:
            # Pega a maior caixa detectada como a ROI principal
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in coordenadas_roi]
            maior_caixa_idx = np.argmax(areas)
            x1, y1, x2, y2 = coordenadas_roi[maior_caixa_idx]
            return [int(x1), int(y1), int(x2), int(y2)]
        
        return None

    def _separar_itens_e_divisores(self, deteccoes):
        """
        Separa as detecções em itens e divisores.
        """
        todos_itens = []
        todos_divisores = []
        for box in deteccoes:
            x1, y1, x2, y2 = box.xyxy[0]
            if int(box.cls) == 0: # classe 0 é 'item'
                todos_itens.append([int(x1), int(y1), int(x2), int(y2)])
            elif int(box.cls) == 1: # classe 1 é 'divisor'
                todos_divisores.append([int(x1), int(y1), int(x2), int(y2)])
        return todos_itens, todos_divisores

    def _filtrar_objetos_na_roi(self, objetos, roi):
        """
        Filtra os objetos que estão dentro da ROI.
        """
        rx1, ry1, rx2, ry2 = roi
        objetos_na_roi = []
        for obj_box in objetos:
            ox1, oy1, ox2, oy2 = obj_box
            # Usa o centro do objeto para verificar se está na ROI
            centro_x = (ox1 + ox2) / 2
            centro_y = (oy1 + oy2) / 2
            if rx1 < centro_x < rx2 and ry1 < centro_y < ry2:
                objetos_na_roi.append(obj_box)
        return objetos_na_roi

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
        "total_camadas": 2
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