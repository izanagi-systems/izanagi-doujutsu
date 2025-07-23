import cv2
import numpy as np
import os

# Desabilita o sync da ultralytics para evitar downloads
os.environ['ULTRALYTICS_SYNC'] = 'False'

# Importa todas as configurações e constantes
from config import *
from detector import Detector
from state_manager import StateManager
from visualizer import Visualizer

class SiacApp:
    """Classe principal que orquestra o sistema SIAC."""
    def __init__(self):
        print("[INFO] Carregando módulos...")
        self.detector = Detector()
        self.state_manager = StateManager()
        self.visualizer = Visualizer()
        print("[INFO] Módulos carregados.")

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Erro ao abrir a fonte de vídeo: {video_source}")
            return

        print("[INFO] Iniciando processamento. Pressione 'q' para sair.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_processado = self.processar_frame(frame)
            cv2.imshow('SIAC - Verificador de Caixas', frame_processado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Processamento finalizado.")

    def processar_frame(self, frame):
        frame_desenhado = frame.copy()

        # 1. Realizar detecção de todos os objetos
        resultados = self.detector.detectar_objetos(frame)
        todos_itens = resultados['itens']
        todos_divisores = resultados['divisores']
        rois_detectadas = resultados['caixas']

        # 2. Encontrar a ROI de maior área para ser a ROI ativa
        roi_ativa = self._get_roi_maior_area(rois_detectadas)

        # 3. Filtrar objetos que estão dentro da ROI ativa
        itens_na_roi = []
        divisores_na_roi = []
        if roi_ativa:
            itens_na_roi = self._filtrar_objetos_na_roi(todos_itens, roi_ativa)
            divisores_na_roi = self._filtrar_objetos_na_roi(todos_divisores, roi_ativa)

        # 4. Atualizar a máquina de estados com as detecções atuais
        self.state_manager.atualizar_estado(roi_ativa, itens_na_roi, divisores_na_roi)

        # 5. Obter o status REAL do sistema para a visualização
        status_visual = self.state_manager.get_status_visual()

        # 6. Desenhar as visualizações usando o Visualizer
        self.visualizer.desenhar_visualizacoes(
            frame_desenhado, 
            roi_ativa, 
            itens_na_roi, 
            divisores_na_roi, 
            status_visual
        )

        return frame_desenhado

    def _get_roi_maior_area(self, rois):
        """De uma lista de ROIs, retorna a que tiver a maior área."""
        if not rois:
            return None
        
        areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rois]
        maior_roi_idx = np.argmax(areas)
        return rois[maior_roi_idx]

    def _filtrar_objetos_na_roi(self, objetos, roi):
        """Filtra uma lista de objetos, retornando apenas os que estão dentro da ROI."""
        objetos_filtrados = []
        rx1, ry1, rx2, ry2 = roi
        for obj in objetos:
            ox1, oy1, ox2, oy2 = obj
            # Verifica se o centro do objeto está dentro da ROI
            centro_x, centro_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            if rx1 < centro_x < rx2 and ry1 < centro_y < ry2:
                objetos_filtrados.append(obj)
        return objetos_filtrados

if __name__ == '__main__':
    app = SiacApp()
    app.run(video_source=0) # Use 0 para webcam ou 'caminho/para/video.mp4'