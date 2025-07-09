from ultralytics import YOLO

def treinar_modelo():
    """
    Carrega um modelo YOLO pré-treinado e inicia o treinamento
    com nosso dataset customizado.
    """
    # Carrega um modelo pré-treinado. 'yolov8n.pt' é o menor e mais rápido.
    # Usar um modelo pré-treinado (transfer learning) acelera muito o processo.
    model = YOLO('yolov8n.pt')

    # Inicia o treinamento
    print("Iniciando o treinamento do modelo. Isso pode levar alguns minutos...")
    results = model.train(
        data=r'c:\Users\ti-005\Desktop\pvcf_gpt\dataset.yaml', 
        epochs=100, 
        imgsz=640, 
        name='contador_itens_aug', 
        # Parâmetros de aumentação de dados
        degrees=25.0,      # Rotação aleatória em graus (-25 a +25)
        translate=0.1,     # Translação aleatória da imagem
        scale=0.1,         # Zoom aleatório na imagem
        fliplr=0.5,        # 50% de chance de espelhar horizontalmente
        mosaic=1.0,        # Usar composição de mosaico
        mixup=0.1          # Misturar imagens (MixUp)
    )
    
    print("\nTreinamento concluído!")
    print(f"O modelo e os resultados foram salvos em: {results.save_dir}")

if __name__ == '__main__':
    treinar_modelo()