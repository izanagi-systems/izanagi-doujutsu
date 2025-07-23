from ultralytics import YOLO
from config import MODELOS, CONFIDENCIA_LIMITE

class Detector:
    """
    Encapsula a lógica de detecção de objetos com os modelos YOLO.
    """
    def __init__(self):
        """
        Carrega os modelos de detecção de ROI e de itens.
        """
        print("[INFO] Carregando modelos de detecção...")
        self.roi_model = YOLO(MODELOS['roi_detector'])
        self.item_model = YOLO(MODELOS['item_detector'])
        print("[INFO] Modelos de detecção carregados.")

    def detectar_objetos(self, frame):
        """
        Executa a detecção de ROI (caixa) e de itens/divisores no frame.

        Args:
            frame: O frame do vídeo a ser processado.

        Returns:
            Um dicionário contendo as listas de bounding boxes para cada classe:
            {'caixas': [], 'itens': [], 'divisores': []}
        """
        # 1. Detectar a ROI (caixas)
        deteccoes_roi = self.roi_model.predict(source=frame, conf=CONFIDENCIA_LIMITE, verbose=False)[0]
        caixas_detectadas = [list(map(int, box.xyxy[0].tolist())) for box in deteccoes_roi.boxes]

        # 2. Detectar Itens e Divisores
        deteccoes_itens = self.item_model.predict(source=frame, conf=CONFIDENCIA_LIMITE, verbose=False)[0]
        itens_detectados = []
        divisores_detectados = []

        for box in deteccoes_itens.boxes:
            coords = list(map(int, box.xyxy[0].tolist()))
            # Classe 0 é 'item', Classe 1 é 'divisor'
            if int(box.cls) == 0:
                itens_detectados.append(coords)
            elif int(box.cls) == 1:
                divisores_detectados.append(coords)

        return {
            'caixas': caixas_detectadas,
            'itens': itens_detectados,
            'divisores': divisores_detectados
        }