import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import deque

class ContadorDeItens:
    def __init__(self, video_source=0, item_model_path='runs/detect/contador_itens_aug2/weights/best.pt', roi_model_path='path/to/roi_model.pt', perfil_caixa=None, timeout_alarme=50):
        """
        Inicializa o contador de itens.
        :param video_source: Fonte do vídeo (0 para webcam, ou caminho para arquivo).
        :param item_model_path: Caminho para o modelo YOLO treinado para contar itens.
        :param roi_model_path: Caminho para o modelo YOLO treinado para detectar a caixa principal (ROI).
        :param perfil_caixa: Dicionário com o perfil da caixa (nome, itens_esperados).
        :param timeout_alarme: Frames para esperar antes de alarmar.
        """
        print(f"[INFO] Carregando modelos...")
        self.item_model = YOLO(item_model_path)
        self.roi_model = YOLO(roi_model_path)
        print(f"[INFO] Modelos carregados com sucesso.")

        self.video_source = video_source
        self.perfil_caixa = perfil_caixa
        if self.perfil_caixa:
            print(f"[INFO] Perfil carregado: {self.perfil_caixa['nome']} (Esperando {self.perfil_caixa['itens_esperados']} itens)")
        else:
            print("[AVISO] Nenhum perfil de caixa carregado. O sistema funcionará em modo de contagem simples.")

        # Detecção do tipo de fonte
        self.is_image = False
        self.is_live = False
        self.cap = None

        if isinstance(video_source, int):
            self.is_live = True
        elif isinstance(video_source, str):
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if any(video_source.lower().endswith(ext) for ext in img_extensions):
                self.is_image = True
        
        if not self.is_image:
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise IOError(f"Não foi possível acessar a fonte de vídeo: {video_source}.")

        # Variáveis para o rastreamento da ROI
        self.caixa_roi = None # (x, y, w, h) da caixa atual

        # Variáveis para o ciclo de vida e alarme
        self.status_caixa = "PROCURANDO"    # Estados: PROCURANDO, ATIVA, PERDIDA
        self.caixa_ativa = None             # A ROI confirmada e que está sendo monitorada
        self.frames_sem_caixa = 0           # Contador de frames desde que a caixa foi perdida
        self.roi_candidata = None           # Uma ROI detectada que pode se tornar a caixa_ativa
        self.contador_frames_estaveis = 0   # Contador para confirmar se uma ROI é estável

        # Constantes de configuração do ciclo de vida
        self.FRAMES_PARA_CONFIRMAR_ROI = 10 # Uma ROI precisa estar estável por 10 frames
        self.FRAMES_DE_CARENCIA = 30        # O operador tem 30 frames para a caixa voltar antes do alarme

        # NOVO: Histórico das últimas contagens para robustez
        self.TAMANHO_HISTORICO = 15 # Armazena as últimas 15 contagens
        self.historico_contagens = deque(maxlen=self.TAMANHO_HISTORICO)

    def _gerenciar_ciclo_caixa(self, frame):
        """
        Gerencia a máquina de estados do ciclo de vida da caixa.
        Retorna a contagem atual e a ROI a ser desenhada.
        """
        contagem_atual = 0
        roi_para_desenhar = None

        # Tenta detectar uma ROI em qualquer estado, para sabermos se a caixa está visível
        roi_detectada_neste_frame = self._detectar_caixa_roi(frame)

        # --- MÁQUINA DE ESTADOS ---
        if self.status_caixa == "PROCURANDO":
            roi_para_desenhar = roi_detectada_neste_frame
            if roi_detectada_neste_frame:
                self.contador_frames_estaveis += 1
                if self.contador_frames_estaveis >= self.FRAMES_PARA_CONFIRMAR_ROI:
                    self.caixa_ativa = roi_detectada_neste_frame
                    self.status_caixa = "ATIVA"
                    print(f"\n[STATUS] Caixa ATIVA. Monitorando conteúdo.")
                    self.historico_contagens.clear() # Limpa o histórico para a nova caixa
            else:
                self.contador_frames_estaveis = 0 # Zera se a caixa sumir

        elif self.status_caixa == "ATIVA":
            if roi_detectada_neste_frame:
                self.caixa_ativa = roi_detectada_neste_frame # Atualiza a posição da ROI ativa
                roi_para_desenhar = self.caixa_ativa
                self.frames_sem_caixa = 0 # Zera o contador de ausência

                # OTIMIZAÇÃO: Só executa a contagem de itens quando a caixa está ativa
                results = self.item_model(frame, verbose=False)
                # CORREÇÃO: Passa a ROI ativa para a função de contagem
                contagem_frame_atual = self._contar_itens_na_roi(results[0], self.caixa_ativa)
                self.historico_contagens.append(contagem_frame_atual) # Adiciona contagem ao histórico
                
                # A contagem exibida é a máxima do histórico recente para estabilidade visual
                contagem_atual = max(self.historico_contagens, default=0)

            else:
                # A caixa sumiu! Inicia o período de carência.
                self.status_caixa = "PERDIDA"
                self.frames_sem_caixa = 1
                print(f"\n[STATUS] Caixa PERDIDA. Iniciando timer de carência...")

        elif self.status_caixa == "PERDIDA":
            if roi_detectada_neste_frame:
                # A caixa voltou a tempo!
                self.status_caixa = "ATIVA"
                self.frames_sem_caixa = 0
                print(f"\n[STATUS] Caixa re-detectada. Voltando a monitorar.")
            else:
                self.frames_sem_caixa += 1
                if self.frames_sem_caixa > self.FRAMES_DE_CARENCIA:
                    # Fim do timeout. Avalia a contagem e dispara o alarme se necessário.
                    contagem_final_confiavel = max(self.historico_contagens, default=0)
                    
                    if contagem_final_confiavel < self.perfil_caixa['itens_esperados']:
                        print(f"\n[ALARME] CAIXA REMOVIDA INCOMPLETA!")
                        print(f"    -> Itens contados: {contagem_final_confiavel} / {self.perfil_caixa['itens_esperados']}")
                    else:
                        print(f"\n[INFO] Caixa removida COMPLETA. Tudo certo.")
                    
                    # Resetar tudo e voltar a procurar
                    self.status_caixa = "PROCURANDO"
                    self.caixa_ativa = None
                    self.ultima_contagem_valida = 0
                    self.frames_sem_caixa = 0
                    self.contador_frames_estaveis = 0

        return contagem_atual, roi_para_desenhar

    def _detectar_caixa_roi(self, frame):
        """
        Detecta a caixa principal (ROI) usando um modelo YOLO dedicado.
        Retorna a bounding box (x, y, w, h) da primeira caixa encontrada ou None.
        """
        # Assume que a classe da caixa principal é 0
        results = self.roi_model(frame, verbose=False, classes=[0], conf=0.6)

        for result in results:
            if len(result.boxes) > 0:
                # Pega a caixa com a maior confiança
                best_box = result.boxes[0]
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                return (x1, y1, w, h)
        
        return None

    def _contar_itens_na_roi(self, result, caixa_roi):
        """
        Filtra e conta as detecções do YOLO que estão dentro da ROI da caixa.
        """
        if caixa_roi is None:
            return 0 # Se não há ROI, não há itens para contar dentro dela.

        contagem_na_roi = 0
        cx, cy, cw, ch = caixa_roi

        for box in result.boxes:
            # Pega o centro da caixa de detecção do item
            item_x1, item_y1, item_x2, item_y2 = box.xyxy[0]
            centro_x = (item_x1 + item_x2) / 2
            centro_y = (item_y1 + item_y2) / 2

            # Verifica se o centro do item está dentro da ROI
            if cx < centro_x < cx + cw and cy < centro_y < cy + ch:
                contagem_na_roi += 1
        
        return contagem_na_roi

    def _loop_video_file(self):
        """
        Loop de processamento para arquivos de vídeo com controles de player.
        Espaço: Play/Pause. 'd'/'a': Frame a frame (quando pausado). 'q': Sair.
        """
        paused = True
        ret, frame = self.cap.read()

        while True:
            if ret: # Só processa e exibe se o frame for válido
                # --- LÓGICA DE DETECÇÃO ---
                contagem, roi_desenho = self._gerenciar_ciclo_caixa(frame)
                
                # --- LÓGICA DE VISUALIZAÇÃO ---
                cv2.putText(frame, f"Contagem: {contagem}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"Status: {self.status_caixa}", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if roi_desenho:
                    x, y, w, h = roi_desenho
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                    cv2.putText(frame, "ROI Caixa", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                cv2.imshow("Contador de Itens - YOLO", frame)
            else:
                print("[INFO] Fim do vídeo. Pressione 'q' para sair ou 'a' para reiniciar.")
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

    def _loop_live_camera(self):
        """
        Loop de processamento para câmera ao vivo.
        Pressione 'q' para sair.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERRO] Não foi possível ler o frame da câmera. Encerrando.")
                break

            # --- LÓGICA DE DETECÇÃO ---
            contagem, roi_desenho = self._gerenciar_ciclo_caixa(frame)

            # --- LÓGICA DE VISUALIZAÇÃO ---
            cv2.putText(frame, f"Contagem: {contagem}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Status: {self.status_caixa}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if roi_desenho:
                x, y, w, h = roi_desenho
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(frame, "ROI Caixa", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            cv2.imshow("Contador de Itens - YOLO", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _processar_imagem_unica(self):
        """
        Processa uma única imagem estática.
        """
        print("[INFO] Iniciando modo Imagem Estática.")
        frame = cv2.imread(self.video_source)
        if frame is None:
            print(f"[ERRO] Não foi possível ler o arquivo de imagem: {self.video_source}")
            return

        # --- LÓGICA DE DETECÇÃO ---
        # O modo de imagem estática não usa o ciclo de vida, apenas a detecção simples.
        caixa_roi_detectada = self._detectar_caixa_roi(frame)
        results = self.item_model(frame, verbose=False)
        result = results[0]
        frame_vis = result.plot()
        
        contagem = self._contar_itens_na_roi(result, caixa_roi_detectada)

        # --- LÓGICA DE VISUALIZAÇÃO ---
        cv2.putText(frame_vis, f"Contagem: {contagem}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if caixa_roi_detectada:
            x, y, w, h = caixa_roi_detectada
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame_vis, "ROI Caixa", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        cv2.imshow("Contador de Itens - YOLO", frame_vis)
        print("[INFO] Pressione qualquer tecla para fechar a imagem.")
        cv2.waitKey(0) # Espera indefinidamente por uma tecla

    def run(self):
        """
        Inicia o processamento, escolhendo o loop apropriado (vídeo, câmera ou imagem).
        """
        if self.is_image:
            self._processar_imagem_unica()
        elif self.is_live:
            print("[INFO] Iniciando modo Câmera ao Vivo. Pressione 'q' para sair.")
            self._loop_live_camera()
        else:
            print("[INFO] Iniciando modo Arquivo de Vídeo.")
            self._loop_video_file()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# Execução do sistema
if __name__ == "__main__":
    # --- CONFIGURAÇÕES ---
    USAR_CAMERA = True
    CAMINHO_VIDEO = "caminho/para/seu/video.mp4"
    CAMINHO_IMAGEM = "caminho/para/sua/imagem.jpg"
    MODELO_ITENS = 'runs/detect/contador_itens_aug2/weights/best.pt'
    MODELO_ROI = 'runs/detect/roi_detector_final2/weights/best.pt' # CAMINHO CORRIGIDO

    # Perfil da caixa a ser verificado
    perfil = {
        'nome': 'Caixa Padrão', 
        'itens_esperados': 12
    }
    # -------------------- #

    if USAR_CAMERA:
        source = 0
    else:
        source = CAMINHO_VIDEO

    contador = ContadorDeItens(video_source=source, 
                               item_model_path=MODELO_ITENS, 
                               roi_model_path=MODELO_ROI, 
                               perfil_caixa=perfil)
    contador.run()
