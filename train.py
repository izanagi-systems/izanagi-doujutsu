from ultralytics import YOLO
import argparse
import os

def treinar_modelo(dataset_name, epochs, imgsz, project_name):
    """
    Carrega um modelo YOLO pré-treinado e inicia o treinamento
    com o dataset customizado especificado.

    :param dataset_name: Nome da pasta do dataset (ex: '1_item_counter').
    :param epochs: Número de épocas para o treinamento.
    :param imgsz: Tamanho da imagem para o treinamento.
    :param project_name: Nome do projeto para salvar os resultados em 'runs/detect/'.
    """
    # Constrói o caminho para o arquivo data.yaml dinamicamente
    base_path = os.getcwd() # Pega o diretório atual
    data_path = os.path.join(base_path, 'dataset', dataset_name, 'data.yaml')

    if not os.path.exists(data_path):
        print(f"[ERRO] Arquivo de configuração não encontrado em: {data_path}")
        print(f"Verifique se o nome do dataset '{dataset_name}' está correto.")
        return

    # Carrega um modelo pré-treinado. 'yolov8n.pt' é o menor e mais rápido.
    model = YOLO('yolov8n.pt')

    # Inicia o treinamento com os parâmetros recebidos
    print(f"-- Iniciando treinamento para o dataset: {dataset_name} --")
    print(f"-- Configuração: {epochs} épocas, tamanho da imagem {imgsz} --")
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=imgsz, 
        name=project_name, 
        # Parâmetros de aumentação de dados (mantidos para robustez)
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
    parser = argparse.ArgumentParser(description="Script de Treinamento YOLOv8 Flexível")
    
    parser.add_argument('--dataset', type=str, required=True, 
                        help="Nome da pasta do dataset a ser usado (ex: '1_item_counter' ou '2_roi_detector').")
    
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Número de épocas de treinamento.")

    parser.add_argument('--imgsz', type=int, default=640, 
                        help="Tamanho da imagem (altura e largura) para o treinamento.")

    parser.add_argument('--name', type=str, default='yolo_treinamento', 
                        help="Nome do projeto/pasta onde os resultados serão salvos.")

    args = parser.parse_args()

    # Se o nome do projeto não for especificado, cria um nome padrão
    project_name = args.name
    if project_name == 'yolo_treinamento':
        project_name = f"{args.dataset}_model"

    treinar_modelo(args.dataset, args.epochs, args.imgsz, project_name)